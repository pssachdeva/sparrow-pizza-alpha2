#!/usr/bin/env python3
from __future__ import annotations

import os
import sys


def main() -> None:
    os.execvp("uv", ["uv", "run", "law-norms-llms-run-async", *sys.argv[1:]])


if __name__ == "__main__":
    main()
