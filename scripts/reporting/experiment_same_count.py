#!/usr/bin/env python3
"""Print RULER/LongBench task metrics on identical completed prompt/seed pairs."""
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from scripts.reporting.experiment_task_breakdown import main


if __name__ == '__main__':
    main(same_count=True)
