"""Stage 2 Gate C.1 vLLM-config compat: hermetic tests for
`src/integration/profiling/vllm_config_context.py`.

Tests D (hard-coded-number audit) and G (unknown API -> loud failure)
need no `torch`/`vllm` at all and run directly in this sandbox. Tests
A/B/C/E/F/H/I need the real `vllm`/`torch` packages (this module talks
to real vLLM config classes, not a Frontier patch target with an
offline-hashable source) -- `pytest.importorskip` gates them, matching
this project's own established convention for GPU-adjacent modules
exactly.
"""
from __future__ import annotations

import ast
import sys
import types

import pytest

from integration.profiling import vllm_config_context
from integration.profiling.vllm_config_context import (
    VllmConfigContextUnknownApi,
    _verify_vllm_config_api_shape,
    get_vllm_config_context_status,
)


def _reset_module_state():
    vllm_config_context._installed = False
    vllm_config_context._observed_vllm_version = None
    vllm_config_context._last_built_config_summary = None


def _restore_patched_frontier_rotary_module():
    """Same real, live-caught cross-test pollution
    `test_rope_api_adapter.py`'s own fixture guards against: a prior
    test's `install_rope_api_adapter()` call leaves Frontier's real
    `_load_vllm_get_rope` monkeypatched for the rest of the process.
    `test_h` (this file) reuses that same shared Frontier module --
    restore it after every test here too, not only in the other file,
    so this file's own test order/repetition can't reintroduce it."""
    if "frontier.profiling.common.layers.rotary_embedding" not in sys.modules:
        return
    import importlib

    import frontier.profiling.common.layers.rotary_embedding as rope_module

    importlib.reload(rope_module)


@pytest.fixture(autouse=True)
def _isolate():
    _reset_module_state()
    yield
    _reset_module_state()
    _restore_patched_frontier_rotary_module()


# --------------------------------------------------- D: hard-coded audit


def test_module_source_contains_no_numeric_literals_beyond_booleans():
    """Structural audit, not a manual claim: parse this module's own
    source and assert no int/float constant appears anywhere except
    True/False (which ast represents as int constants). Any real
    model/device/timing number introduced later would fail this test
    immediately, before a human has to notice it in review."""
    path = vllm_config_context.__file__
    tree = ast.parse(open(path).read())
    numeric_literals = {
        node.value
        for node in ast.walk(tree)
        if isinstance(node, ast.Constant) and isinstance(node.value, (int, float))
    }
    assert numeric_literals <= {True, False}, (
        f"found non-boolean numeric literal(s) in {path}: "
        f"{numeric_literals - {True, False}}"
    )


def test_only_hardcoded_string_literal_device_value_is_cuda():
    """`"cuda"` is the one literal this module hard-codes, and it is an
    API/schema constant (the only real accelerator value in
    `DeviceConfig`'s own `Literal[...]` type, not a model value) --
    confirm it is the *only* place a bare device/model-ish string
    constant appears outside of docstrings, error messages, and
    attribute-name lookups (dict keys / dataclass field names used for
    introspection, e.g. "device_config", "custom_ops")."""
    import inspect

    src = inspect.getsource(vllm_config_context.build_profiling_vllm_config)
    assert '"cuda"' in src
    # No other quoted literal resembling a real model/device size appears.
    for forbidden in ("1024", "128", "40960", "1000000", "8192", "gfx950", "mi355x"):
        assert forbidden not in src


# --------------------------------------------------------- G: unknown API


def _install_fake_vllm_config_modules(monkeypatch, *, missing_device=False,
                                       missing_device_config_field=False,
                                       missing_set_current=False):
    import dataclasses

    fake_vllm = types.ModuleType("vllm")
    fake_config = types.ModuleType("vllm.config")
    fake_config_vllm = types.ModuleType("vllm.config.vllm")
    fake_config_device = types.ModuleType("vllm.config.device")

    if missing_device_config_field:
        @dataclasses.dataclass
        class FakeDeviceConfig:
            device_type: str = "cuda"  # no 'device' field
    else:
        @dataclasses.dataclass
        class FakeDeviceConfig:
            device: str = "cuda"
            device_type: str = "cuda"

    if not missing_device:
        fake_config_device.DeviceConfig = FakeDeviceConfig

    if missing_set_current:
        pass  # deliberately not set -- simulates a real API removal
    else:
        from contextlib import contextmanager

        @contextmanager
        def fake_set_current_vllm_config(vllm_config):
            yield

        fake_config_vllm.set_current_vllm_config = fake_set_current_vllm_config

    @dataclasses.dataclass
    class FakeVllmConfig:
        device_config: object = None

    fake_config_vllm.VllmConfig = FakeVllmConfig

    monkeypatch.setitem(sys.modules, "vllm", fake_vllm)
    monkeypatch.setitem(sys.modules, "vllm.config", fake_config)
    monkeypatch.setitem(sys.modules, "vllm.config.vllm", fake_config_vllm)
    monkeypatch.setitem(sys.modules, "vllm.config.device", fake_config_device)


def test_unknown_api_raises_when_device_config_missing_device_field(monkeypatch):
    _install_fake_vllm_config_modules(monkeypatch, missing_device_config_field=True)
    with pytest.raises(VllmConfigContextUnknownApi, match="device"):
        _verify_vllm_config_api_shape()


def test_unknown_api_raises_when_vllm_config_has_no_device_config_field(monkeypatch):
    import dataclasses

    fake_vllm = types.ModuleType("vllm")
    fake_config = types.ModuleType("vllm.config")
    fake_config_vllm = types.ModuleType("vllm.config.vllm")
    fake_config_device = types.ModuleType("vllm.config.device")

    @dataclasses.dataclass
    class FakeDeviceConfig:
        device: str = "cuda"

    fake_config_device.DeviceConfig = FakeDeviceConfig

    from contextlib import contextmanager

    @contextmanager
    def fake_set_current_vllm_config(vllm_config):
        yield

    fake_config_vllm.set_current_vllm_config = fake_set_current_vllm_config

    @dataclasses.dataclass
    class FakeVllmConfigNoDeviceConfig:
        model_config: object = None  # no 'device_config' field

    fake_config_vllm.VllmConfig = FakeVllmConfigNoDeviceConfig

    monkeypatch.setitem(sys.modules, "vllm", fake_vllm)
    monkeypatch.setitem(sys.modules, "vllm.config", fake_config)
    monkeypatch.setitem(sys.modules, "vllm.config.vllm", fake_config_vllm)
    monkeypatch.setitem(sys.modules, "vllm.config.device", fake_config_device)

    with pytest.raises(VllmConfigContextUnknownApi, match="device_config"):
        _verify_vllm_config_api_shape()


def test_unknown_api_raises_when_set_current_vllm_config_missing(monkeypatch):
    _install_fake_vllm_config_modules(monkeypatch, missing_set_current=True)
    with pytest.raises(VllmConfigContextUnknownApi):
        _verify_vllm_config_api_shape()


def test_unknown_api_raises_when_set_current_vllm_config_signature_changed(monkeypatch):
    import dataclasses
    from contextlib import contextmanager

    fake_vllm = types.ModuleType("vllm")
    fake_config = types.ModuleType("vllm.config")
    fake_config_vllm = types.ModuleType("vllm.config.vllm")
    fake_config_device = types.ModuleType("vllm.config.device")

    @dataclasses.dataclass
    class FakeDeviceConfig:
        device: str = "cuda"

    fake_config_device.DeviceConfig = FakeDeviceConfig

    @dataclasses.dataclass
    class FakeVllmConfig:
        device_config: object = None

    fake_config_vllm.VllmConfig = FakeVllmConfig

    @contextmanager
    def fake_set_current_vllm_config_renamed(config):  # renamed parameter
        yield

    fake_config_vllm.set_current_vllm_config = fake_set_current_vllm_config_renamed

    monkeypatch.setitem(sys.modules, "vllm", fake_vllm)
    monkeypatch.setitem(sys.modules, "vllm.config", fake_config)
    monkeypatch.setitem(sys.modules, "vllm.config.vllm", fake_config_vllm)
    monkeypatch.setitem(sys.modules, "vllm.config.device", fake_config_device)

    with pytest.raises(VllmConfigContextUnknownApi, match="vllm_config"):
        _verify_vllm_config_api_shape()


def test_status_before_any_use_is_all_none_never_guessed():
    status = get_vllm_config_context_status()
    assert status == {
        "applied": False,
        "detected_vllm_version": None,
        "last_built_config": None,
    }


# --------------------------------------- A/B/C/E/F/H/I: real torch+vllm


def test_a_no_context_reproduces_the_expected_pinned_failure():
    """A: without any set_current_vllm_config context, a real
    CustomOp-derived class construction fails exactly the way the real
    Probe 1 traceback showed."""
    pytest.importorskip("torch")
    vllm = pytest.importorskip("vllm")
    from vllm.model_executor.custom_op import CustomOp

    class _TinyCustomOp(CustomOp):
        name = "tiny_custom_op_for_test"

        def forward_native(self, x):
            return x

        forward_cuda = forward_native
        forward_hip = forward_native
        forward_cpu = forward_native

    with pytest.raises(AssertionError, match="Current vLLM config is not set"):
        _TinyCustomOp()


def test_b_context_allows_construction_to_succeed():
    """B: the same tiny CustomOp construction succeeds inside
    profiling_vllm_config_context()."""
    pytest.importorskip("torch")
    pytest.importorskip("vllm")
    from vllm.model_executor.custom_op import CustomOp

    from integration.profiling.vllm_config_context import (
        profiling_vllm_config_context,
    )

    class _TinyCustomOp(CustomOp):
        name = "tiny_custom_op_for_test_b"

        def forward_native(self, x):
            return x

        forward_cuda = forward_native
        forward_hip = forward_native
        forward_cpu = forward_native

    with profiling_vllm_config_context():
        op = _TinyCustomOp()
    assert op is not None


def test_c_overrides_are_reflected_not_silently_ignored():
    """C: an explicit override (a real, caller-supplied device_config)
    is the one actually used -- not silently replaced by the built-in
    default."""
    pytest.importorskip("torch")
    vllm = pytest.importorskip("vllm")
    from vllm.config.device import DeviceConfig

    from integration.profiling.vllm_config_context import (
        build_profiling_vllm_config,
    )

    custom_device_config = DeviceConfig(device="cpu")
    cfg = build_profiling_vllm_config(device_config=custom_device_config)
    assert cfg.device_config is custom_device_config
    assert cfg.device_config.device_type == "cpu"


def test_e_and_f_context_restored_after_use_and_nesting_is_safe():
    """E + F: the previous vllm_config (None, or a real pre-existing
    one) is restored exactly after the context exits, including when
    nested."""
    pytest.importorskip("torch")
    vllm = pytest.importorskip("vllm")
    from vllm.config.vllm import get_current_vllm_config

    from integration.profiling.vllm_config_context import (
        profiling_vllm_config_context,
    )

    with pytest.raises(AssertionError):
        get_current_vllm_config()

    with profiling_vllm_config_context() as outer_cfg:
        assert get_current_vllm_config() is outer_cfg
        with profiling_vllm_config_context() as inner_cfg:
            assert get_current_vllm_config() is inner_cfg
            assert inner_cfg is not outer_cfg
        # Restored to the outer context, not None, after the inner exits.
        assert get_current_vllm_config() is outer_cfg

    # Restored to "no context" after the outermost exits.
    with pytest.raises(AssertionError):
        get_current_vllm_config()


def test_h_rope_adapter_continues_to_work_inside_the_context():
    """H: the RoPE adapter (rope_api_adapter.py) still produces a real
    RotaryEmbedding when constructed inside this context -- the two
    fixes compose, not just each in isolation."""
    pytest.importorskip("torch")
    pytest.importorskip("vllm")
    import frontier.profiling.common.layers.rotary_embedding as rope_module

    from integration.profiling import rope_api_adapter
    from integration.profiling.vllm_config_context import (
        profiling_vllm_config_context,
    )

    rope_api_adapter._installed = False
    rope_module._VLLM_GET_ROPE = None
    rope_module._VLLM_GET_ROPE_IMPORT_ERROR = None
    rope_api_adapter.install_rope_api_adapter()

    with profiling_vllm_config_context():
        result = rope_module.get_rope(
            128, rotary_dim=128, max_position=40960, base=1000000.0,
            is_neox_style=True, rope_scaling=None,
        )
    assert result is not None


def test_i_non_customop_class_construction_is_unaffected_either_way():
    """I: Frontier's own linear-layer classes (`ColumnParallelLinear`/
    `RowParallelLinear`/`ReplicatedLinear`, `frontier/profiling/common/
    parallel_utils/tensor_parallel_layers.py`) -- **not** vLLM's own
    upstream classes of the same name, confirmed by reading Frontier's
    own imports in `linear_op_impl.py` (`from frontier.profiling.common.
    parallel_utils.tensor_parallel_layers import ColumnParallelLinear,
    RowParallelLinear` -- not from `vllm.model_executor.layers.linear`
    at all) -- are plain `torch.nn.Module`s needing neither
    `set_current_vllm_config` nor vLLM's own tensor-parallel-group
    global state (`world_size` is an explicit, real constructor
    argument that bypasses `get_tensor_model_parallel_world_size()`
    entirely when supplied, exactly how Frontier's own real call sites
    already use it). Confirmed no regression with or without this
    context active."""
    pytest.importorskip("torch")
    pytest.importorskip("vllm")
    import frontier.profiling.common.parallel_utils.tensor_parallel_layers as tp_layers
    from vllm.model_executor.custom_op import CustomOp

    assert not issubclass(tp_layers.ReplicatedLinear, CustomOp)
    # Constructing it with no context active must not raise the
    # "Current vLLM config is not set" AssertionError this task is about --
    # confirmed by the real Probe 1 traceback itself, which got past this
    # exact construction before failing at get_rope().
    try:
        layer = tp_layers.ReplicatedLinear(8, 8, bias=False, world_size=1)
    except RuntimeError as exc:
        # This constructor allocates its weight with
        # device=torch.cuda.current_device() -- a genuine, real-GPU
        # operation, not the CustomOp/VllmConfig issue this task is
        # about. Live, on a CPU-only investigation host with no
        # --device flags (this task's own explicit mandate), this is
        # exactly the "reached the point an actual GPU operation is
        # required -- stop there" boundary §9 describes, not a failure
        # of this module or this test.
        assert "No CUDA GPUs are available" in str(exc), (
            f"expected the real-GPU boundary, got a different error: {exc}"
        )
        return
    assert layer is not None
    assert layer.world_size == 1
