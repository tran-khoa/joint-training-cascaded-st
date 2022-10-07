import argparse
import csv
from tqdm import tqdm

parser = argparse.ArgumentParser()
parser.add_argument('seg')
parser.add_argument('tsv')
parser.add_argument('out')
parser.add_argument('--efficient', action='store_true')
args = parser.parse_args()

print('Reading segments...')
with open(args.seg) as f:
    segs = f.read().splitlines()
    if not args.seg.endswith('.seg'):
        segs = list(x.split("/") for x in segs)
        segs = set(f"{x[1]}_{x[2]}" for x in segs)

print('Reading tsv...')
with open(args.tsv) as f:
    reader = csv.DictReader(
        f,
        delimiter="\t",
        quotechar=None,
        doublequote=False,
        lineterminator="\n",
        quoting=csv.QUOTE_NONE,
    )
    samples = [dict(e) for e in reader]

kept = []
filtered = []
unmapped = set(segs)

for sample in tqdm(samples):
    query = sample['id']
    if query in segs:
        kept.append(sample)
        if not args.efficient and query in unmapped:
            unmapped.remove(query)
    else:
        if not args.efficient:
            filtered.append(sample)

print('Storing...')
with open(args.out, 'wt') as f:
    writer = csv.DictWriter(f,
                            fieldnames=reader.fieldnames,
                            delimiter='\t',
                            quotechar=None,
                            doublequote=False,
                            lineterminator="\n",
                            escapechar="\\",
                            quoting=csv.QUOTE_NONE)
    writer.writeheader()
    writer.writerows(kept)

if args.efficient:
    print(f'Done, kept {len(kept)} samples')
else:
    print(f'Done, kept {len(kept)} samples, filtered {len(filtered)} samples.')
    print(f'{len(unmapped)} segments were unmapped, these were')
    for _id in unmapped:
        print(_id)