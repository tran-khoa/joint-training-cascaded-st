import argparse
from nltk import sent_tokenize
from tqdm import tqdm

parser = argparse.ArgumentParser()
parser.add_argument('inp')
parser.add_argument('out')
args = parser.parse_args()

with open(args.inp, 'rt') as i_f, open(args.out, 'wt') as o_f:
    for line in tqdm(i_f.readlines()):
        sentences = (s[0].upper() + s[1:] for s in sent_tokenize(line.strip(), language='german'))
        o_f.write(" ".join(sentences) + '\n')
