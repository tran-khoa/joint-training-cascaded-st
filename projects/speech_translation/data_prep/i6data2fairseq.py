import argparse
import csv
import gzip
import hashlib
import pickle
import re
import shutil
from pathlib import Path
from typing import Union, Any, Sequence, Hashable

import numpy as np
import pandas as pd
from lxml import etree
from lxml.etree import ElementTree
from tqdm import tqdm
import signal
from returnn_archiver import open_file_archive

MANIFEST_COLUMNS = ["id", "audio", "n_frames", "start", "end", "tgt_text"]

PATH_REGEX = re.compile(r"corpus/(.*)/(.*)")


class DelayedKeyboardInterrupt:

    def __enter__(self):
        self.signal_received = False
        self.old_handler = signal.signal(signal.SIGINT, self.handler)

    def handler(self, sig, frame):
        self.signal_received = (sig, frame)

    def __exit__(self, type, value, traceback):
        signal.signal(signal.SIGINT, self.old_handler)
        if self.signal_received:
            self.old_handler(*self.signal_received)


def save_df_to_tsv(dataframe, path: Union[str, Path], columns: Union[Sequence[Hashable], None], header: bool):
    _path = path if isinstance(path, str) else path.as_posix()
    dataframe.to_csv(
        _path,
        sep="\t",
        header=header,
        index=False,
        encoding="utf-8",
        escapechar="\\",
        columns=columns,
        quoting=csv.QUOTE_NONE,
        float_format="%.2f"
    )


def merge_dicts_recursively(target_dict: dict[str, Any], src_dict: dict[str, Any]):
    # in-place!
    for k, v in src_dict.items():
        if k not in target_dict or not isinstance(v, dict):
            target_dict[k] = v
        else:
            merge_dicts_recursively(target_dict[k], v)

    return target_dict


def create_manifest(results: dict[str, dict[str, dict[str, Union[str, int]]]], output_path: Path) -> None:
    # results to manifest
    manifest = {c: [] for c in MANIFEST_COLUMNS}
    for rec_id, sgmt in results.items():
        for seg_id, data in sgmt.items():
            manifest['id'].append(data['id'])
            manifest['audio'].append(data['audio'])
            manifest["n_frames"].append(data['n_frames'])
            manifest["start"].append(data['start'])
            manifest["end"].append(data['end'])
            manifest["tgt_text"].append(data['tgt_text'])
    df = pd.DataFrame.from_dict(manifest)

    save_df_to_tsv(df, output_path.as_posix(), columns=MANIFEST_COLUMNS, header=False)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('features', help='path to .bundle file (list of feature paths)')
    parser.add_argument('corpus', help='path to .xml file')
    parser.add_argument('target')
    parser.add_argument('--translation-pkl')
    parser.add_argument('--translation-corpus')
    parser.add_argument('--segments')
    parser.add_argument('--split', default='train', help='should start with train, dev or test')

    args = parser.parse_args()

    target_path = Path(args.target)
    target_path.mkdir(exist_ok=True, parents=True)
    (target_path / 'fbank80').mkdir(exist_ok=True)

    logfile_path = (target_path / 'i6data2fairseq.log')
    cache_path = (target_path / 'cache')
    cache_path.mkdir(exist_ok=True)

    segments = None
    if args.segments:
        segments = {}

        with open(args.segments, 'rt') as f:
            lines = f.read().splitlines()

        num_segments = len(lines)
        for line in lines:
            _, recording, segment = line.split('/')
            if recording not in segments:
                segments[recording] = set()
            segments[recording].add(segment)

        del lines
        print(f'Loaded {args.segments}, found {num_segments} for {len(segments)} recordings.')
    else:
        print('No segments loaded.')

    corpus = {}

    if (target_path / 'corpus_cache.pkl').exists():
        print('Loading corpus from cache, skipping step 1.')
        with open(target_path / 'corpus_cache.pkl', 'rb') as f:
            corpus = pickle.load(f)
    else:
        print('STEP 1. Extracting corpus from xml...')
        assert not ((args.translation_pkl is not None) ^ (args.translation_corpus is not None))
        if args.translation_pkl is not None:
            with open(args.translation_pkl, 'rb') as f:
                translation_meta = pickle.load(f)

            fopen = gzip.open if args.translation_corpus.endswith('.gz') else open
            with fopen(args.translation_corpus, 'rt') as f:
                translations = f.readlines()

        original_tree: ElementTree = etree.parse(args.corpus)
        original_tree.getroot()
        for recording in tqdm(original_tree.getroot().getchildren()):
            if recording.tag != 'recording':
                print(f'Invalid element in corpus: {etree.tostring(recording, pretty_print=True)}')

            recording_id = recording.attrib['name']
            if segments and recording_id not in segments:
                continue

            for _s_id, segment in enumerate(recording.getchildren()):
                segment_id = segment.attrib['name'] if 'name' in segment.attrib else str(_s_id + 1)
                if segments and segment_id not in segments[recording_id]:
                    continue

                if recording_id not in corpus:
                    corpus[recording_id] = {}
                if segment_id in corpus[recording_id]:
                    print(f"WARNING: Segment {segment_id} of recording {recording_id} is duplicate in the corpus xml!")

                text = ""
                if args.translation_pkl is not None:
                    key = f'corpus/{recording_id}/{segment_id}'
                    assert key in translation_meta['sprint'], f'FATAL Error: Could not find key {key} in translation pickle'
                    idx = translation_meta['sprint'].index(key)
                    line = int(translation_meta['translation'][idx].lstrip('line-`'))
                    text = translations[line]
                else:
                    for orth in segment.getchildren():
                        text += orth.text

                text = text.replace("@@ ", "")

                corpus[recording_id][segment_id] = {
                    'start': float(segment.attrib.get('start', -1)),
                    'end': float(segment.attrib.get('end', -1)),
                    'tgt_text': text.strip()
                }
        with open(target_path / 'corpus_cache.pkl', 'wb') as f:
            pickle.dump(corpus, f)

    print('STEP 2. Extracting features, creating manifests...')
    with open(args.features, 'rt') as f:
        paths = [line.strip() for line in f.readlines()]

    num_xml_missing = 0
    total_segments = 0
    pbar = tqdm(paths)

    with logfile_path.open('wt') as logfile:
        for path in pbar:
            hashing = hashlib.sha256()
            hashing.update(path.encode())
            hex_hash = hashing.hexdigest()
            cache_file = (cache_path / hex_hash).with_suffix('.tsv')

            if cache_file.exists():
                continue

            bundle_results = {}

            fa_p = open_file_archive(path)

            inner_pbar = tqdm(fa_p.file_list())

            for file_id in inner_pbar:
                if file_id.endswith('.attribs'):
                    continue
                total_segments += 1
                pbar.set_description(f"{Path(path).stem}, segments: {total_segments}")

                matches = re.match(PATH_REGEX, file_id)
                if matches is None:
                    logfile.write(f'WARNING: What is {file_id} in {path}?')
                    continue
                recording_id, segment_id = matches.group(1), matches.group(2)
                uid = f"{recording_id}_{segment_id}"
                inner_pbar.set_description(f"Rec {recording_id}, Seg {segment_id}")

                if segments:
                    if recording_id not in segments:
                        continue
                    if segment_id not in segments[recording_id]:
                        continue

                if recording_id not in corpus or segment_id not in corpus[recording_id]:
                    logfile.write(f'WARNING: Could not find {recording_id}, segment {segment_id} in corpus xml (path {path}, skipping...\n')
                    continue

                feature_path = target_path / 'fbank80' / uid
                feature_path = feature_path.with_suffix(feature_path.suffix + '.npy')
                try:
                    _, feature_list = fa_p.read(file_id, 'feat')  # list of 80-dim features
                except Exception as e:
                    logfile.write(f'WARNING: Could not read {file_id} in {path}. Please check if this is desired!\n')
                    logfile.write(str(e) + '\n')
                    continue

                if not feature_path.exists():
                    features = np.array(feature_list)  # (T x D)
                    with DelayedKeyboardInterrupt():
                        np.save(feature_path.as_posix(), features)

                if recording_id not in bundle_results:
                    bundle_results[recording_id] = {}

                bundle_results[recording_id][segment_id] = {
                    'id': uid,
                    'audio': feature_path.as_posix(),
                    'n_frames': len(feature_list),
                    'start': corpus[recording_id][segment_id]['start'],
                    'end': corpus[recording_id][segment_id]['end'],
                    'tgt_text': corpus[recording_id][segment_id]['tgt_text']
                }

            with DelayedKeyboardInterrupt():
                create_manifest(bundle_results, cache_file)

    print('STEP 3. Merging manifests...')
    with (target_path / args.split).with_suffix('.tsv').open('wt') as outfile:
        # Write the header
        outfile.write("\t".join(MANIFEST_COLUMNS) + "\n")

        for file in tqdm(list((target_path / cache_path).glob("*.tsv"))):
            with file.open('rt') as infile:
                shutil.copyfileobj(infile, outfile)

    print('Done.')
