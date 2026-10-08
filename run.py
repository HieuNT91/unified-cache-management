#!/usr/bin/env python3
"""Prepare prompts or a persistent KV collection, then run a native clean method."""
import argparse
from pathlib import Path

from runner.config import METHODS
from runner.preparation import prepare
from runner.single import run


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest='command', required=True)
    prep = sub.add_parser('prepare', help='CPU tokenization; no truncation')
    prep.add_argument('--model', type=Path, required=True)
    prep.add_argument('--context', type=Path, required=True)
    prep.add_argument('--question', type=Path, required=True)
    prep.add_argument('--output', type=Path, required=True)
    prep.add_argument('--thinking', action=argparse.BooleanOptionalAction, default=True)
    prep.add_argument('--max-output-tokens', type=int, default=16384)
    prep.set_defaults(func=prepare)
    launch = sub.add_parser('run')
    launch.add_argument('--model', type=Path)
    inputs = launch.add_mutually_exclusive_group(required=True)
    inputs.add_argument('--input', type=Path)
    inputs.add_argument('--setup', type=Path)
    launch.add_argument('--prompt-ids', nargs='+')
    launch.add_argument('--max-output-tokens', type=int)
    launch.add_argument('--output', type=Path, required=True)
    launch.add_argument('--method', choices=METHODS, required=True)
    launch.add_argument('--ratio', type=float)
    launch.add_argument('--router-policy', type=Path)
    launch.add_argument('--router-id', choices=('router1','router2','router3'), default='router1')
    layers = launch.add_mutually_exclusive_group()
    layers.add_argument('--layers', nargs='+', type=int)
    layers.add_argument('--num-layers', type=int)
    launch.add_argument('--tp', type=int)
    launch.add_argument('--memory', type=float, default=.9)
    launch.add_argument('--dry-run', action='store_true', help='Print config without loading GPU libraries')
    launch.set_defaults(func=run)
    setup_parser = sub.add_parser('setup', help='Freeze a collection and construct persistent context KV')
    setup_parser.add_argument('--model', type=Path, required=True)
    setup_parser.add_argument('--manifest', type=Path, required=True)
    setup_parser.add_argument('--output', type=Path, required=True)
    setup_parser.add_argument('--tp', type=int, default=4)
    setup_parser.add_argument('--memory', type=float, default=.9)
    setup_parser.add_argument('--kv-chunk-size', type=int, default=4096)
    setup_parser.add_argument('--context-length', type=int, default=131072)
    setup_parser.add_argument('--thinking', action=argparse.BooleanOptionalAction, default=True)
    setup_parser.add_argument('--max-output-tokens', type=int)
    setup_parser.add_argument('--dry-run', action='store_true')
    from runner.setups import build_setup, run_collection
    setup_parser.set_defaults(func=build_setup)
    args = parser.parse_args()
    if args.model is not None:
        args.model = args.model.resolve()
    if args.command == 'run':
        if args.method != 'router' and args.ratio is None:
            args.ratio = .2
        if args.method == 'router' and (args.ratio is not None or args.layers is not None or args.num_layers is not None):
            parser.error('Router rejects manual ratio and layer overrides')
        if args.setup is not None:
            args.func = run_collection
        else:
            if args.model is None or args.prompt_ids is not None:
                parser.error('--input requires --model and does not accept --prompt-ids')
            args.tp = 4 if args.tp is None else args.tp
    args.func(args)


if __name__ == '__main__':
    main()
