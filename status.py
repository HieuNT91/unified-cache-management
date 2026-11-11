#!/usr/bin/env python3
"""Report committed AdaTree/no-cache results, overall and by dataset subtask."""
import argparse
import os
from pathlib import Path

from runner.reporting.run import report

ROOT = Path(__file__).resolve().parent


def runs(path):
    """Find run metadata, stopping before descending into each run's artifacts."""
    path = Path(path).resolve()
    if '.trash' in path.parts:
        raise ValueError('Archived directories are outside this report scope')
    if not path.is_dir():
        raise ValueError('Results directory does not exist: ' + str(path))
    found = []
    for folder, directories, files in os.walk(path, followlinks=False):
        directories[:] = sorted(name for name in directories if name != '.trash'
                                and not (Path(folder) / name).is_symlink())
        if 'settings.json' in files:
            found.append(Path(folder))
            directories[:] = []
    if not found:
        raise ValueError('No configured runs found under ' + str(path))
    return found


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('results', nargs='?', type=Path, default=ROOT / 'results',
                        help='Run directory or results directory (default: this checkout/results)')
    parser.add_argument('--mode', choices=('full', 'same-input'), default='full',
                        help='All available results, or exact shared inputs/seeds (default: full)')
    args = parser.parse_args(argv)
    try:
        for index, base in enumerate(runs(args.results)):
            text, _, _ = report(base, mode=args.mode)
            if index:
                print()
            print(text, end='')
    except (OSError, ValueError, KeyError, TypeError) as error:
        parser.exit(1, f'Cannot report {args.results}: {error}\n')


if __name__ == '__main__':
    main()
