import argparse

parser = argparse.ArgumentParser()
parser.add_argument('corpus')
parser.add_argument('meta')
parser.add_argument('out_pref')
parser.add_argument('--beam', type=int, default=12)
parser.add_argument('--is-mt')
args = parser.parse_args()

print('Reading corpus...')
with open(args.corpus, 'rt') as f:
    corpus = f.read().splitlines()

print("Reading meta...")
with open(args.meta, 'rt') as f:
    meta = [l.split() for l in f.read().splitlines()]
    if not args.is_mt:
        meta = [(int(t[0]), float(t[1])) for t in meta]
    else:
        meta = [(int(t[0]) // 12, float(t[1])) for t in meta]  # TODO

result = {}

print("Summarizing...")
for (line_id, score), txt in zip(meta, corpus):
    if line_id not in result:
        result[line_id] = []

    result[line_id].append(
        {
            "text": txt,
            "score": score
        }
    )

for k in result.keys():
    result[k].sort(key=lambda x: x['score'], reverse=True)

print("Writing to output...")
running_idx = 0
with open(f"{args.out_pref}_corpus", 'wt') as out_corpus, open(f"{args.out_pref}_meta", 'wt') as out_meta:
    for line_id in sorted(result.keys()):
        score_sum = sum(x['score'] for x in result[line_id][:args.beam])
        for num, hyp in enumerate(result[line_id]):
            out_corpus.write(hyp['text'] + '\n')

            if not args.is_mt:
                _id = line_id
            else:
                _id = running_idx
                running_idx += 1

            score = hyp['score']
            if num >= args.beam:
                score = 0
            out_meta.write(f"{_id} {score / score_sum} {hyp['score']}\n")
