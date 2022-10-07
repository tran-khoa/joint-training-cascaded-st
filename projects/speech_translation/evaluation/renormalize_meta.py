import argparse
import math


parser = argparse.ArgumentParser()
parser.add_argument('meta')
parser.add_argument('corpus')
parser.add_argument('meta_new')
parser.add_argument('corpus_new')
args = parser.parse_args()

with open(args.meta, 'rt') as f:
    meta = f.read().splitlines()

with open(args.corpus, 'rt') as f:
    corpus = f.read().splitlines()

print("Files read.")

current_id = None
current_total = 0

top_score, top_hyp = -math.inf, None

with open(args.meta_new, 'wt') as f_meta_new, open(args.corpus_new, 'wt') as f_corpus_new:
    for line, txt in zip(meta, corpus):
        _id, score, _ = line.split()
        _id = int(_id)
        score = float(score)

        if _id != current_id:
            if current_id is not None:
                f_meta_new.write(f"{current_id} {top_score / current_total} 42\n")
                f_corpus_new.write(f"{top_hyp}\n")

            current_id = _id
            current_total = score
            top_score, top_hyp = score, txt
        else:
            if score > top_score:
                top_score, top_hyp = score, txt
            current_total += score

    f_meta_new.write(f"{current_id} {top_score / current_total} 42\n")
    f_corpus_new.write(f"{top_hyp}\n")
