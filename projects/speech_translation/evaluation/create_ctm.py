#!/usr/bin/env python3
import argparse
import re
from operator import itemgetter

p = argparse.ArgumentParser()
p.add_argument('infile')
p.add_argument('outfile')
p.add_argument('--stm')
args = p.parse_args()


with open(args.stm, 'rt') as f:
    segments = list({'idx': e.split()[0], 'start': float(e.split()[3]), 'end': float(e.split()[4])}
                    for e in f.readlines())

with open(args.infile, 'rt') as infile:
    hyps = list(filter(lambda s: s.startswith('H-'), infile.readlines()))
    hyps = list(re.sub(r'^H-', '', s) for s in hyps)
    hyps = list(s.split('\t') for s in hyps)
    hyps = list((int(idx), text.replace(" ", "").replace("▁", " ").strip()) for idx, _, text in hyps)
    hyps = list(sorted(hyps, key=itemgetter(0)))
    hyps = list(text for _, text in hyps)

with open(args.outfile, 'wt') as outfile:
    outfile.write(";; <name> <track> <start> <duration> <word> <confidence> [<n-best>]\n")
    for idx, hyp_line in enumerate(hyps):
        outfile.write(f";; {segments[idx]['idx']} ({segments[idx]['start']}-{segments[idx]['end']})\n")
        if hyp_line:
            words = hyp_line.split()
            avg_dur = (segments[idx]['end'] - segments[idx]['start']) * 0.9 / max(len(words), 1)

            for i in range(len(words)):
                if '^;' in words[i] or '[UNK]' in words[i]:
                    continue
                outfile.write(f"{segments[idx]['idx']} 1 {segments[idx]['start'] + i * avg_dur} {avg_dur} {words[i]} 0.99\n")
    outfile.write("}\n")
