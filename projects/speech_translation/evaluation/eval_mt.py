import argparse
import os

parser = argparse.ArgumentParser()
parser.add_argument('dataset', choices=['tstCOMMON', 'tstHE'])
parser.add_argument('hyp')
args = parser.parse_args()

dataset = args.dataset
hyp = args.hyp

with open(f'/work/bahar/st/iwslt2020/data/final_data/seg_{dataset}', 'rt') as f:
    segs = f.read().splitlines()

with open(hyp, 'rt') as f:
    hyps = f.read().splitlines()
    hyps = filter(lambda x: x.startswith('H-'), hyps)
    hyps = [h.split('\t') for h in hyps]
    hyps = [(int(h[0].replace('H-','')), h[2]) for h in hyps]
    hyps = sorted(hyps, key=lambda x: x[0])
    hyps = [h[1] for h in hyps]

with open(f'hyp_{dataset}.txt', 'wt') as f:
    for seg, hyp in zip(segs, hyps):
        f.write(f"{seg}: {hyp}\n")

os.system(f'/work/bahar/st/iwslt2020/data/final_data/ref/mt/mteval.tercom.sh {dataset} hyp_{dataset}.txt')
