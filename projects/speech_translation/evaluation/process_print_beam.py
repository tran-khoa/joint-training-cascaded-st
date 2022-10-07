import argparse
import pickle
import sacrebleu
from tqdm import tqdm

parser = argparse.ArgumentParser()
parser.add_argument("path")
parser.add_argument("target")
parser.add_argument("cache")
args = parser.parse_args()

result = {}

with open(args.target, "rt") as f:
    target = f.read().splitlines()

with open(args.path, "rt") as f:
    for line in tqdm(f):
        line = line.strip()
        if line.startswith("TlHyp-") or line.startswith("TsHyp-"):
            pref, score, txt = line.split('\t')
            _id, _beam = pref.split("-")[-1].split(".")
            _id = int(_id)
            _beam = int(_beam)

            if _id not in result:
                result[_id] = {}

            if _beam not in result[_id]:
                result[_id][_beam] = {"target": target[_id]}

            if line.startswith("TlHyp-"):
                result[_id][_beam]["translation"] = txt
                result[_id][_beam]["translation_score"] = float(score.split("|")[0])
                result[_id][_beam]["translation_bleu"] = sacrebleu.sentence_bleu(txt, [target[_id]]).score
                result[_id][_beam]["translation_bp"] = sacrebleu.sentence_bleu(txt, [target[_id]]).bp
                result[_id][_beam]["translation_ter"] = sacrebleu.sentence_ter(txt, [target[_id]]).score
            else:
                result[_id][_beam]["transcript"] = txt
                result[_id][_beam]["transcript_score"] = float(score)

with open(args.cache, "wb") as f:
    pickle.dump(result, f)
