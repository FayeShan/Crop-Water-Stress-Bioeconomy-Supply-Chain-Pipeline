#!/usr/bin/env python
"""
Run GLORIA MRIO Analysis

Usage:
    python run.py --config configs/default.json
    python run.py --config configs/default.json --years 1990 1991 1992
    python run.py --config configs/default.json --targets Gold Agriculture
"""

import argparse
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from gloria_mrio import GloriaMRIO, Config


def main():
    parser = argparse.ArgumentParser(description='Run GLORIA MRIO Analysis')
    parser.add_argument('--config', type=str, required=True,
                        help='Path to config JSON file')
    parser.add_argument('--years', type=int, nargs='+', default=None,
                        help='Override years to process')
    parser.add_argument('--targets', type=str, nargs='+', default=None,
                        help='Override target sector names')
    parser.add_argument('--indicators', type=int, nargs='+', default=None,
                        help='Override indicators to calculate')
    
    args = parser.parse_args()

    print(f"Loading config from: {args.config}")
    config = Config.from_json(args.config)

    if args.years:
        config.years = args.years

    analyzer = GloriaMRIO(config)
    results = analyzer.run(
        target_names=args.targets,
        indicators=args.indicators
    )
    
    return 0 if results else 1


if __name__ == "__main__":
    sys.exit(main())
