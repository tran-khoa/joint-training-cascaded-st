import argparse
import itertools

import re

annotations = {"(Applaus)", "(Lachen)", "(vorgetäuschtes Schluchzen)"}

parser = argparse.ArgumentParser()
parser.add_argument('sub')
parser.add_argument('sup')
parser.add_argument('out')
args = parser.parse_args()

with open(args.sub, 'rt') as sub, open(args.sup, 'rt') as sup:
    super_dict = {re.sub(r" *", "", a.strip().lower()): a.strip() for a in sup.readlines() if a.strip()}
    subset = list(a.strip() for a in sub.readlines())

"""
similars = {}
for el, val in (super_dict.items()):
    annotations = list(re.finditer(r"\(.*?\)|\[.*?\]", el))
    combinations = list(itertools.combinations(annotations, i + 1) for i in range(len(annotations)))
    combinations = list(z for y in combinations for z in y)

    def apply_combination(txt, comb):
        selectors = [1] * len(txt)
        for match_obj in comb:
            selectors[match_obj.span()[0]: match_obj.span()[1]] = [0] * (match_obj.span()[1] - match_obj.span()[0])
        txt = ''.join(itertools.compress(txt, selectors))
        txt = re.sub(" +", " ", txt).strip()
        txt = re.sub(r" \.", "", txt)
        return txt


    keys = [apply_combination(el, c) for c in combinations]
    vals = [apply_combination(val, c) for c in combinations]

    for key, orig in zip(keys, vals):
        similars[key] = orig
   

super_dict.update(similars)
 """
with open(args.out, 'wt') as f:
    for el in subset:
        query = re.sub(r"  *", "", el.strip().lower())
        #query = re.sub(r"\. \"$", ".\"", query)
        #query = re.sub(r"\. \"", ".\" ", query)

        if query in super_dict:
            res = super_dict[query]
            f.write(res + '\n')
        else:
            print('Could not align: ' + query)
            f.write("| TODO\n")
