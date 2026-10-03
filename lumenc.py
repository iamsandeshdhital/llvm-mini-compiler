#!/usr/bin/env python3
"""Run the Lumen compiler straight from a source checkout.

Lets a fresh clone work without installing anything or setting PYTHONPATH::

    python lumenc.py examples/hello.lum --run-ir

Installing the package (``pip install -e .``) provides the same driver as the
``lumenc`` command.
"""

from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "src"))

from lumen.driver import main  # noqa: E402  (needs the path insertion above)

if __name__ == "__main__":
    sys.exit(main())