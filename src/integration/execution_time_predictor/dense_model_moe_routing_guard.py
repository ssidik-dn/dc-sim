"""Stage 2 Gate C.1: `SklearnDisaggregationExecutionTimePredictor.__init__`
(`frontier/execution_time_predictor/sklearn_disaggregation_execution_time_predictor.py`)
computes MoE expert-routing allocations unconditionally for every
predictor whose `cluster_type` is `PREFILL`/`DECODE_FFN`/`DECODE`,
regardless of `model_config.is_moe` -- confirmed a real, narrow gap, not
an architectural incompatibility (docs/tasks/68-stage2-gate-c1-dense-routing-report.md).

Every real *consumer* of the routing state this computes
(`_calculate_expert_token_allocation`, expert-parallel communication) is
already gated on `model_config.is_moe` -- confirmed by reading every call
site (docs/tasks/68 S2): none is reachable when `is_moe` is `False`.
Computing routing for a dense model is dead work whose only effect is to
crash on `total_expert_num=0` (`1.0 / total_expert_num` in
`_generate_expert_allocations`). Frontier's own predictor-selection
function (`random_forrest_execution_time_predictor.py::_get_base_class`)
already branches dense-vs-MoE via `model_config.is_moe` -- but only in
non-disaggregated mode; this gap is that same check never having been
propagated into disaggregated mode's own eager routing pre-computation.

This patch reuses the exact `is_moe_model = model_config is not None and
model_config.is_moe` idiom this same Frontier file already applies three
other places (`_get_dummy_execution_time_for_cluster` and the
communication-metadata construction that follows it), rather than
inventing new semantics.

Two real, distinct states, both handled explicitly:

- `is_moe=False` (any `total_expert_num`): a dense model. Routing
  computation is skipped entirely -- `self._prefill_routing_details` /
  `_decode_ffn_routing_details` / `_decode_routing_details` end up
  deleted (absent), exactly the way the pre-existing `DECODE_ATTN`
  branch already leaves them for the same reason ("not needed"). No
  synthetic expert, no empty-but-present placeholder dict is fabricated
  -- there is no consumer that would ever read one (docs/tasks/68 S2/S6).
- `is_moe=True` with `total_expert_num<=0`: **inconsistent model
  metadata**, not a legitimate dense model -- the model's own config
  says it is MoE but declares no experts. Raising
  `InconsistentMoeModelMetadataError` here, explicitly, replaces relying
  on the incidental `ZeroDivisionError` a few calls further down the
  same original path would otherwise still produce (the amendment to
  docs/tasks/68's original proposal, approved before this module was
  written) -- a clearer failure at the actual point of inconsistency,
  not a behavior change to which models are rejected.
- `is_moe=True` with `total_expert_num>0`: the original, pre-existing
  code executes character-for-character unchanged (docs/tasks/68 S12).

Guarded by a source hash over the whole `__init__`
(`SklearnDisaggregationExecutionTimePredictor.__init__`), following
`..replica_scheduler.sglang_guard`'s (task 47) and
`.mla_phase_filter`'s (task 53) established pattern of patching a whole
pinned-Frontier method, guarded, rather than editing the checkout.
"""
from __future__ import annotations

import hashlib
import inspect
from typing import Dict, Optional

from frontier.config import (
    BaseExecutionTimePredictorConfig,
    BaseReplicaSchedulerConfig,
    ClusterConfig,
    MetricsConfig,
    ReplicaConfig,
)
from frontier.execution_time_predictor.shared_prediction_model_manager import (
    ExecutionTimePredictionModelManager,
)
from frontier.execution_time_predictor.sklearn_disaggregation_execution_time_predictor import (
    SklearnDisaggregationExecutionTimePredictor,
)
from frontier.types import ClusterType

# Computed against the checked-out Frontier's own
# SklearnDisaggregationExecutionTimePredictor.__init__
# (frontier/execution_time_predictor/sklearn_disaggregation_execution_time_predictor.py)
# at the time this module was written. A changed hash means the method's
# own body changed upstream -- install_dense_model_moe_routing_guard()
# raises rather than patch over an unknown implementation.
_EXPECTED_SOURCE_HASH = "bc5e32d80eecdfcb06af26968b577fb7d4015adf32e0a509fb7ee1b98065c099"

_installed = False


class DenseModelMoeRoutingGuardMismatch(RuntimeError):
    pass


class InconsistentMoeModelMetadataError(ValueError):
    pass


def _patched_init(
    self,
    predictor_config: BaseExecutionTimePredictorConfig,
    replica_config: ReplicaConfig,  # This is a representative config
    replica_scheduler_config: BaseReplicaSchedulerConfig,
    metrics_config: MetricsConfig,
    cluster_config: ClusterConfig = None,
    model_manager: ExecutionTimePredictionModelManager = None,
    cluster_type: ClusterType = None,
    training_file_paths: Dict[str, str] = None,
    actual_replica_ids: Optional[list] = None,
    cc_backend: Optional[object] = None,
) -> None:
    # We still call super() with one of the configs to set up the basic models.
    # The prefill config is a good representative as it's a full model.
    super(SklearnDisaggregationExecutionTimePredictor, self).__init__(
        predictor_config,
        replica_config,
        replica_scheduler_config,
        metrics_config,
        model_manager,
        cluster_type,
        training_file_paths,
        cc_backend,
    )

    assert (
        cluster_config is not None
    ), "cluster_config cannot be None for SklearnDisaggregationExecutionTimePredictor"
    self._cluster_config = cluster_config

    # Store actual replica ids if provided (to align routing_details keys with cluster replica IDs)
    self._actual_replica_ids = actual_replica_ids

    # Override MoE parameters with cluster-specific values
    # The parent class uses the representative replica_config, but we need cluster-specific configs
    self._cluster_type = cluster_type
    cluster_replica_config = replica_config
    if cluster_type:
        cluster_replica_config = self._get_cluster_replica_config(cluster_type)
        # Override MoE parameters for this specific cluster
        self._moe_ep_size = cluster_replica_config.moe_expert_parallel_size
        self._moe_tp_size = cluster_replica_config.moe_tensor_parallel_size
        self._router_topk = cluster_replica_config.router_topk

    self._workload_distribution_type = self._resolve_workload_distribution_type(
        getattr(
            cluster_replica_config,
            "moe_routing_distribution_type",
            "balanced",
        )
    )
    # Use moe_routing_seed from config for deterministic routing simulation.
    self._distribution_seed = getattr(cluster_replica_config, "moe_routing_seed", 42)

    if (
        not hasattr(self._cluster_config, "prefill_replica_config")
        or self._cluster_config.prefill_replica_config is None
    ):

        if (
            hasattr(self._cluster_config, "replica_config")
            and self._cluster_config.replica_config is not None
        ):
            self._cluster_config.prefill_replica_config = (
                self._cluster_config.replica_config
            )
            self._cluster_config.decode_ffn_replica_config = (
                self._cluster_config.replica_config
            )
            if not hasattr(self._cluster_config, "prefill_cluster_num_replicas"):
                self._cluster_config.prefill_cluster_num_replicas = getattr(
                    self._cluster_config, "num_replicas", 1
                )
            if not hasattr(self._cluster_config, "decode_ffn_cluster_num_replicas"):
                self._cluster_config.decode_ffn_cluster_num_replicas = getattr(
                    self._cluster_config, "num_replicas", 1
                )
        else:
            raise ValueError(
                "Neither prefill_replica_config nor replica_config is available in cluster_config"
            )

    # Pre-calculate routing details only for relevant clusters to avoid unnecessary computation
    # Each predictor only calculates routing for clusters it will actually serve

    self._prefill_routing_details = None
    self._decode_ffn_routing_details = None
    self._decode_routing_details = None  # For unified DECODE cluster in PD-disaggregation mode

    # Define cluster types that require MoE routing details
    # DECODE is included for PD-disaggregation mode where DECODE handles both attention + MoE
    moe_cluster_types = {ClusterType.PREFILL, ClusterType.DECODE_FFN, ClusterType.DECODE}
    current_cluster_types = {cluster_type} if cluster_type else moe_cluster_types

    # Stage 2 Gate C.1 (docs/tasks/68): a dense model (is_moe=False) has no
    # experts to route. Every real consumer of routing_details
    # (_calculate_expert_token_allocation, expert-parallel communication)
    # already gates itself on model_config.is_moe, not on whether this
    # state happens to be present -- so computing it for a dense model is
    # dead work whose only effect is to crash on total_expert_num=0.
    # `is_moe_model` reuses the exact idiom this same Frontier file
    # already applies elsewhere (_get_dummy_execution_time_for_cluster /
    # the communication-metadata construction just below it).
    is_moe_model = self._model_config is not None and self._model_config.is_moe

    if is_moe_model:
        # Calculate routing details for each relevant cluster type
        for target_cluster_type in current_cluster_types.intersection(
            moe_cluster_types
        ):
            # Stage 2 Gate C.1 amendment: is_moe=True with
            # total_expert_num<=0 is inconsistent model metadata, not a
            # legitimate dense model -- fail explicitly here, at the
            # actual point of inconsistency, instead of relying on the
            # incidental ZeroDivisionError `_generate_expert_allocations`
            # would otherwise raise a few calls further down the same
            # path. Checked per target_cluster_type's OWN replica config
            # (not the constructed predictor's own `cluster_replica_config`,
            # which for a DECODE_ATTN predictor legitimately carries no
            # expert count regardless of whether the model overall is
            # MoE -- DECODE_ATTN never reaches this loop at all, since it
            # is not a member of moe_cluster_types, so that case can never
            # trigger this check; this loop only ever visits PREFILL/
            # DECODE_FFN/DECODE, each of which genuinely needs experts
            # when the model is MoE).
            target_replica_config = self._get_cluster_replica_config(target_cluster_type)
            total_expert_num_for_cluster = getattr(
                target_replica_config, "total_expert_num", None
            )
            if total_expert_num_for_cluster is not None and total_expert_num_for_cluster <= 0:
                raise InconsistentMoeModelMetadataError(
                    "Model config declares is_moe=True but "
                    f"total_expert_num={total_expert_num_for_cluster} "
                    f"(cluster_type={target_cluster_type}) -- inconsistent "
                    "MoE model metadata; refusing to simulate expert "
                    "routing for a model that claims to be MoE while "
                    "declaring no experts."
                )

            routing_details: Dict[int, Dict[int, Dict[int, float]]] = (
                self._simulate_and_store_routing(target_cluster_type)
            )

            if target_cluster_type == ClusterType.PREFILL:
                self._prefill_routing_details = routing_details
                del self._decode_ffn_routing_details
                del self._decode_routing_details
            elif target_cluster_type == ClusterType.DECODE_FFN:
                self._decode_ffn_routing_details = routing_details
                del self._prefill_routing_details
                del self._decode_routing_details
            elif target_cluster_type == ClusterType.DECODE:
                self._decode_routing_details = routing_details
                del self._prefill_routing_details
                del self._decode_ffn_routing_details

    # Initialize empty routing details for clusters/models that don't need MoE routing
    if not is_moe_model or cluster_type == ClusterType.DECODE_ATTN:
        del self._prefill_routing_details
        del self._decode_ffn_routing_details
        del self._decode_routing_details


def install_dense_model_moe_routing_guard() -> None:
    """Patch `SklearnDisaggregationExecutionTimePredictor.__init__` so a
    dense model (`model_config.is_moe=False`) skips MoE expert-routing
    simulation entirely, instead of crashing on `total_expert_num=0`, and
    a model with genuinely inconsistent metadata
    (`is_moe=True, total_expert_num<=0`) fails with an explicit
    `InconsistentMoeModelMetadataError` rather than an incidental
    `ZeroDivisionError`. Safe to call more than once (idempotent). Not
    called by `install()` by default -- a caller must ask for it
    explicitly, the same way every other patch in this project is
    opt-in.

    Raises `DenseModelMoeRoutingGuardMismatch` if
    `SklearnDisaggregationExecutionTimePredictor.__init__`'s source no
    longer matches what this module was written against.
    """
    global _installed
    if _installed:
        return
    current_hash = hashlib.sha256(
        inspect.getsource(
            SklearnDisaggregationExecutionTimePredictor.__init__
        ).encode()
    ).hexdigest()
    if current_hash != _EXPECTED_SOURCE_HASH:
        raise DenseModelMoeRoutingGuardMismatch(
            f"SklearnDisaggregationExecutionTimePredictor.__init__'s source "
            f"has changed (hash {current_hash} != expected "
            f"{_EXPECTED_SOURCE_HASH}). Refusing to install the dense-model "
            f"routing-guard patch over an implementation this project "
            f"hasn't reviewed -- update _EXPECTED_SOURCE_HASH in "
            f"{__name__} only after confirming the routing-computation "
            f"block being guarded is still the same one, and that no "
            f"is_moe-aware guard was added upstream in the meantime.")
    SklearnDisaggregationExecutionTimePredictor.__init__ = _patched_init
    _installed = True
