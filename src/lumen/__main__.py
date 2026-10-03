"""``python -m lumen`` entry point."""

from __future__ import annotations

import sys

from .driver import main

if __name__ == "__main__":
    sys.exit(main())