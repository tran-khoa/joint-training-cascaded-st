#!/usr/bin/env python3

import argparse
import re
from tqdm import tqdm

from returnn_archiver import open_file_archive

PATH_REGEX = re.compile(r"corpus/(.*)/(.*)")


parser = argparse.ArgumentParser()
parser.add_argument('files', type=str, nargs='+')
args = parser.parse_args()

with open('i6_archive_ids.log', 'wt') as logfile:
    for path in args.files:
        fa_p = open_file_archive(path)

        for file_id in fa_p.file_list():
            if file_id.endswith('.attribs'):
                continue

            matches = re.match(PATH_REGEX, file_id)
            if matches is None:
                logfile.write(f'WARNING: What is {file_id} in {path}?')
                continue
            recording_id, segment_id = matches.group(1), matches.group(2)
            uid = f"{recording_id}_{segment_id}"
            print(uid)
