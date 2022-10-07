import argparse
from tqdm import tqdm
import gzip

parser = argparse.ArgumentParser()
parser.add_argument("corpus")
parser.add_argument("--vocab")
args = parser.parse_args()

if args.vocab:
    with open(args.vocab, "rt") as f:
        vocab = set(l.split()[0] for l in f.readlines())


num_lines = 0
num_words = 0
unique_words = set()
unknown_words = list()

_open = gzip.open if args.corpus.endswith(".gz") else open

with _open(args.corpus, "rt") as f:
    for line in tqdm(f):
        for word in line.strip().split():
            num_words += 1
            unique_words.add(word)
            if args.vocab and word not in vocab:
                unknown_words.append(word)

print(f"segments: {num_lines}")
print(f"running (sub?)words: {num_words}")
print(f"(sub?)word vocabulary: {len(unique_words)}")
print(f"running OOV rate: {len(unknown_words)} ({len(unknown_words)/num_words*100:.1f}%)")
