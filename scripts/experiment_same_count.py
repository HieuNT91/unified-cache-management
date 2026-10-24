#!/usr/bin/env python3
"""Print RULER/LongBench task metrics on identical completed prompt/seed pairs."""
from experiment_task_breakdown import main


if __name__ == '__main__':
    main(same_count=True)
