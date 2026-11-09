"""Stdlib fallback test runner, for environments without pytest.

Imports every proto/tests/test_*.py, calls each top-level test_* function, prints a
summary and exits nonzero on any failure. Tests are plain functions using assert, so
they also run unchanged under pytest.
"""

import importlib.util
import os
import sys
import traceback

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)


def main() -> int:
    tests_dir = os.path.join(HERE, "tests")
    failures = 0
    total = 0
    for fname in sorted(os.listdir(tests_dir)):
        if not (fname.startswith("test_") and fname.endswith(".py")):
            continue
        spec = importlib.util.spec_from_file_location(fname[:-3], os.path.join(tests_dir, fname))
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        for name in sorted(dir(mod)):
            if name.startswith("test_") and callable(getattr(mod, name)):
                total += 1
                try:
                    getattr(mod, name)()
                    print(f"PASS {fname}::{name}")
                except Exception:
                    failures += 1
                    print(f"FAIL {fname}::{name}")
                    traceback.print_exc()
    print(f"\n{total - failures}/{total} passed")
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
