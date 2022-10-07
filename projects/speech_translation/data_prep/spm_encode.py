import argparse
import sentencepiece as spm
from tqdm import tqdm

parser = argparse.ArgumentParser()
parser.add_argument('model')
parser.add_argument('raw')
args = parser.parse_args()

sp = spm.SentencePieceProcessor(model_file=args.model)
with open(args.raw, 'rt') as in_file:
    for in_line in tqdm(in_file):
        if in_line.strip() == '<UNK>':
            print("<unk>")
            continue
        out_line = sp.Encode(in_line, out_type='str')
        print(' '.join(out_line))
