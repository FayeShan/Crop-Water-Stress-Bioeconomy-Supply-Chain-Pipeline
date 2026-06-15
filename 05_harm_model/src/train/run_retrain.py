"""HARM final retrain dispatcher: routes to the unweighted or weighted (--weighted) impl file."""

import sys


def main() -> int:
    # Parse only the --weighted flag here; everything else is forwarded.
    argv = list(sys.argv[1:])
    weighted = False
    if "--weighted" in argv:
        argv.remove("--weighted")
        weighted = True

    # Forward remaining args to the relevant impl's argparse via sys.argv
    sys.argv = [sys.argv[0]] + argv

    if weighted:
        from . import _retrain_weighted_impl as impl
    else:
        from . import _retrain_unweighted_impl as impl
    impl.main()
    return 0


if __name__ == "__main__":
    # Allow direct execution: python src/train/run_retrain.py ...
    # (Relative imports require package context, so fall back to
    #  explicit module loading here.)
    import importlib
    import os
    from pathlib import Path

    argv = list(sys.argv[1:])
    weighted = "--weighted" in argv
    if weighted:
        argv.remove("--weighted")
    sys.argv = [sys.argv[0]] + argv

    # Add parent of src/ to path so ``import src.train.*`` works.
    this_dir = Path(__file__).resolve().parent
    root_dir = this_dir.parent.parent  # 05_harm_model/
    if str(root_dir) not in sys.path:
        sys.path.insert(0, str(root_dir))

    impl_name = (
        "src.train._retrain_weighted_impl" if weighted
        else "src.train._retrain_unweighted_impl"
    )
    impl = importlib.import_module(impl_name)
    impl.main()
