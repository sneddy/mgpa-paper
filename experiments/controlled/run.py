#!/usr/bin/env python3
"""Run the published controlled experiment; all numerical code is in mgpa."""
from pathlib import Path
import sys

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[1]))
from experiment import data
from mgpa.data.runner import main

if __name__ == "__main__":
    main(HERE, data)
