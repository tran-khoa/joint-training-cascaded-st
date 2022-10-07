#!/usr/bin/env python3

import argparse

p = argparse.ArgumentParser()
p.add_argument('path')
path = p.parse_args().path

with open(path, 'rt') as f:
    for l in f.readlines():
        print(l.replace(" ", "").replace("\u2581", " ").strip())
