#!/usr/bin/env python
import argparse
import gzip
from tqdm import tqdm
from hashlib import md5
# import heapq


def _hash(x):
    return md5(x.encode()).hexdigest()


def _normalize(x):
    x = x.strip()
    x = x.replace("—", "-")
    return x


parser = argparse.ArgumentParser()
parser.add_argument('sup')
parser.add_argument('sub', nargs="+")
parser.add_argument('--out')
args = parser.parse_args()


superset = dict()
result = list()

print(f'Reading superset {args.sup}...')
if args.sup.endswith('.gz'):
    _open = gzip.open
else:
    _open = open

with _open(args.sup, 'rt') as f:
    for line, text in tqdm(enumerate(f)):
        text = _normalize(text)
        superset[_hash(text)] = line

    for sub in args.sub:
        print(f'Reading subset {sub}...')
        if sub.endswith('.gz'):
            _open = gzip.open
        else:
            _open = open
        with _open(sub, 'rt') as f:
            for line_pos, text in tqdm(enumerate(f)):
                text = _normalize(text)
                text_hash = _hash(text)
                if text_hash in superset:
                    result.append((line_pos, text))

with open(args.out, 'wt') as f:
    for i, (line_pos, text) in enumerate(result):
        f.write(f"{line_pos}\t{text}")
        if i < len(result) - 1:
            f.write('\n')
