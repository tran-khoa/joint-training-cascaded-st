import argparse
import csv
import operator
from pathlib import Path

import yaml
import numpy as np
from tqdm import tqdm

parser = argparse.ArgumentParser()
parser.add_argument('query')
parser.add_argument('key')
parser.add_argument('values')
parser.add_argument('outpath')
args = parser.parse_args()

cwd = Path(args.outpath)
cwd.mkdir(parents=True, exist_ok=True)

print('Reading values...')
with open(args.values, 'rt') as vf:
    values = vf.readlines()

print('Reading keys...')
with open(args.key) as kf:
    key_list = yaml.load(kf, Loader=yaml.CLoader)

keys = {}
print('Grouping...')
for idx, e in tqdm(enumerate(key_list)):
    if e['wav'] not in keys:
        keys[e['wav']] = []
    keys[e['wav']].append(
        (float(e['offset']), float(e['offset']) + float(e['duration']), idx)
    )
print('Sorting...')
for af in keys:
    keys[af].sort(key=lambda x: (x[0], x[1]))

print('Reading query...')
with open(args.query) as f:
    reader = csv.DictReader(
        f,
        delimiter="\t",
        quotechar=None,
        doublequote=False,
        lineterminator="\n",
        quoting=csv.QUOTE_NONE,
    )
    samples = (dict(e) for e in reader)

    print('Grouping...')
    queries = {}
    for sample in samples:
        ted_audioint = int(sample['id'].split('_')[1].split('-')[0])
        ted_audiofile = f"ted_{ted_audioint}.wav"
        assert ted_audiofile in keys
        if ted_audiofile not in queries:
            queries[ted_audiofile] = []

        queries[ted_audiofile].append((float(sample['start']), float(sample['end']), sample))

print('Sorting...')
for af in queries:
    queries[af].sort(key=lambda x: (x[0], x[1]))

print('Aligning...')
alignments = {af: {} for af in queries.keys()}


def segments_equal(_q, _k):
    return np.isclose(_q[0], _k[0]) and np.isclose(_q[1], _k[1])


results = []
for af, query_list in tqdm(queries.items()):
    if (cwd / f'out_{af}.tsv').exists():
        # could do this earlier but nah
        continue

    key_list = keys[af]

    key_idx = 0
    for query_idx, q in tqdm(enumerate(query_list)):
        found = False
        while True:
            if segments_equal(q, key_list[key_idx]):
                found = True
                value_idx = key_list[key_idx][-1]
                q[-1]['src_text'] = q[-1]['tgt_text']
                q[-1]['tgt_text'] = values[value_idx].strip()
                results.append(q[-1])
                break
            else:
                key_idx += 1
            if key_idx >= len(key_list):
                break
        if not found:
            raise ValueError(f'Could not align {af}, segment {q}.')

    print(f'Storing results for {af}...')
    with open(cwd / f'out_{af}.tsv', 'w') as of:
        writer = csv.DictWriter(of,
                                fieldnames=['id', 'audio', 'n_frames', 'start', 'end', 'src_text', 'tgt_text'],
                                delimiter='\t',
                                quotechar=None,
                                doublequote=False,
                                lineterminator="\n",
                                escapechar="\\",
                                quoting=csv.QUOTE_NONE)
        writer.writerows(results)
