#!/usr/bin/env python3
"""Compatibility entry point for the AdaTree launcher."""
from pathlib import Path
import os
import sys

if __name__ == '__main__':
    os.execvp('bash', ['bash', str(Path(__file__).resolve().with_name('run.sh')), *sys.argv[1:]])
