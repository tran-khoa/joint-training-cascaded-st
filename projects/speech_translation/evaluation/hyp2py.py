#!/usr/bin/env python3
import argparse
import re
from operator import itemgetter


p = argparse.ArgumentParser()
p.add_argument('infile')
p.add_argument('stm')
args = p.parse_args()

with open(args.stm, 'rt') as f:
    segments = list(e.split()[0] for e in f.readlines())

with open(args.infile, 'rt') as infile:
    hyps = list(filter(lambda s: s.startswith('H-'), infile.readlines()))
    hyps = list(re.sub(r'^H-', '', s) for s in hyps)
    hyps = list(s.split('\t') for s in hyps)
    hyps = list((int(idx), text.replace(" ", "").replace("▁", " ").strip()) for idx, _, text in hyps)
    hyps = list(sorted(hyps, key=itemgetter(0)))
    hyps = list(text for _, text in hyps)

with open(args.outfile, 'wt') as outfile:
    _res = {_id: hyp for _id, hyp in zip(segments, hyps)}
    print(_res)