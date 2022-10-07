#!/usr/bin/env python3
import argparse
import csv


parser = argparse.ArgumentParser()
parser.add_argument('orig')
parser.add_argument('paths')
args = parser.parse_args()

print('Loading paths...')
paths = {}
with open(args.paths, 'rt') as f:
    for _id, path in (x.split() for x in f.read().splitlines()):
        paths[_id] = path

print(f'Loaded paths for {len(paths)} segments.')

print('Loading original tsv...')
with open(args.orig, 'rt') as f:
    reader = csv.DictReader(
        f,
        delimiter="\t",
        quotechar=None,
        doublequote=False,
        lineterminator="\n",
        quoting=csv.QUOTE_NONE,
    )
    samples = list(dict(e) for e in reader)

print('Changing paths...')
for sample in samples:
    sample['audio'] = paths[sample['id']]

print('Writing tsv...')
with open(args.orig + ".zipped", 'wt') as f:
    writer = csv.DictWriter(
        f,
        fieldnames=reader.fieldnames,
        delimiter="\t",
        quotechar=None,
        doublequote=False,
        lineterminator="\n",
        quoting=csv.QUOTE_NONE,
    )
    writer.writeheader()
    writer.writerows(samples)
print('Done.')
