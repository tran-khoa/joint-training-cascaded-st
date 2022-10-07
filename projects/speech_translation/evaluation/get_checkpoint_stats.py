import argparse
import torch

parser = argparse.ArgumentParser()
parser.add_argument('model')
args = parser.parse_args()

a = torch.load(args.model)
if 'extra_state' in a:
    if 'subepoch' in a['extra_state']:
        print(f"Subepoch: {a['extra_state']['subepoch']}")
    print(f"Epoch: {a['extra_state']['train_iterator']['epoch']}")
else:
    print('Error: Checkpoint has no extra state.')
