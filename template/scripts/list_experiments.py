#!/usr/bin/env python
"""List registered experiments."""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from myproject.registries import ExperimentRegistry
import myproject.experiments  # noqa: F401


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("-v", "--verbose", action="store_true")
    args = parser.parse_args()

    for name, cls in ExperimentRegistry.items():
        if args.verbose:
            doc = (cls.__doc__ or "").strip().split("\n")[0]
            print(f"{name}: {cls.__name__} - {doc}")
        else:
            print(name)


if __name__ == "__main__":
    main()
