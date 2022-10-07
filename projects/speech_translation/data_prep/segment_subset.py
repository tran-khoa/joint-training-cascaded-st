#!/usr/bin/env python3

import argparse
parser = argparse.ArgumentParser()
parser.add_argument('--sub')
parser.add_argument('--sup')
args = parser.parse_args()

with open(args.sup, 'rt') as f:
    sup = set(f.read().splitlines())

with open(args.sub, 'rt') as f:
    for line in f.read().splitlines():
        if line not in sup:
            print(f'NOT a subset, {line} not in potential superset.')
            break
    print('Is a subset.')
