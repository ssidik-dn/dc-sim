#!/usr/bin/env python3
"""Enforce the import direction that keeps the engine portable.

    src/engine/  must never import from  src/integration/  or  upstream/

The engine answers placement and communication-cost questions on its own. If
it starts importing Frontier, the host choice stops being reversible and the
engine can no longer be tested, published, or reused without it.

This is a structural rule, so it is checked structurally: parse the AST rather
than grepping, so a commented-out import or a string mentioning "frontier"
does not trip it.

Exit code 1 on violation. Wire into CI.
"""
from __future__ import annotations

import ast
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
ENGINE = ROOT / "src" / "engine"

FORBIDDEN_PREFIXES = ("integration", "frontier", "astra_sim", "upstream")


def imported_names(path: Path):
    """Yield (module, lineno) for every import in a file."""
    try:
        tree = ast.parse(path.read_text(), filename=str(path))
    except SyntaxError as e:
        print(f"  ! {path}: syntax error: {e}")
        return
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for a in node.names:
                yield a.name, node.lineno
        elif isinstance(node, ast.ImportFrom):
            # level > 0 is a relative import, always inside the engine
            if node.level and node.level > 0:
                continue
            if node.module:
                yield node.module, node.lineno


def main() -> int:
    if not ENGINE.is_dir():
        print(f"engine directory not found: {ENGINE}")
        return 1

    violations = []
    files = sorted(ENGINE.rglob("*.py"))
    for f in files:
        for mod, lineno in imported_names(f):
            head = mod.split(".")[0]
            if head in FORBIDDEN_PREFIXES:
                violations.append((f.relative_to(ROOT), lineno, mod))

    print(f"checked {len(files)} files under src/engine/")
    if violations:
        print("\nIMPORT DIRECTION VIOLATION\n")
        for path, lineno, mod in violations:
            print(f"  {path}:{lineno}  imports {mod!r}")
        print("\nsrc/engine must not depend on src/integration or upstream/.")
        print("Move the code, or invert the dependency behind an interface.")
        return 1

    print("OK: engine imports nothing from integration or upstream")
    return 0


if __name__ == "__main__":
    sys.exit(main())
