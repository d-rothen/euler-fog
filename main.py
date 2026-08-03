"""Thin wrapper so ``python main.py`` works without installing the package.

The logic lives in :mod:`euler_preprocess.cli`, which is also the installed
``euler-preprocess`` console-script entry point.
"""

import sys

from euler_preprocess.cli import main

if __name__ == "__main__":
    sys.exit(main())
