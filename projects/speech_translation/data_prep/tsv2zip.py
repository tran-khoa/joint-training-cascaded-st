# adapted from https://github.com/thecml/pytorch-lmdb/blob/main/folder2lmdb.py
import argparse
import csv
import zipfile
from pathlib import Path
from tqdm import tqdm

parser = argparse.ArgumentParser()
parser.add_argument('tsv')
parser.add_argument('outpath')
args = parser.parse_args()

cwd = Path(args.outpath)
cwd.mkdir(parents=True, exist_ok=True)

with open(args.tsv) as meta_fh:
    reader = csv.DictReader(
        meta_fh,
        delimiter="\t",
        quotechar=None,
        doublequote=False,
        lineterminator="\n",
        quoting=csv.QUOTE_NONE,
    )
    samples = list(dict(e) for e in reader)


out_path = Path(args.outpath)
zip_path = out_path / "zips"
zip_path.mkdir(exist_ok=True, parents=True)

size_limit = int(10e9)  # 10 GBs max (~8 GB actually)
threshold = size_limit * 0.8

curr_zip_path = lambda _idx: zip_path / f"audio_{_idx}.zip"
curr_zip_idx = 0
curr_zip_size = 0

zip_fh = zipfile.ZipFile(curr_zip_path(curr_zip_idx), "w", zipfile.ZIP_STORED)

with open(out_path / "zipped_paths.txt", "at") as meta_fh:
    for idx, s in enumerate(tqdm(samples)):
        feat_path = Path(s["audio"])
        feat_size = feat_path.stat().st_size

        if curr_zip_size + feat_size > threshold:
            print(f"Opening up new database {curr_zip_idx + 1}...")
            curr_zip_size = 0
            curr_zip_idx += 1
            zip_fh = zipfile.ZipFile(curr_zip_path(curr_zip_idx), "w", zipfile.ZIP_STORED)

        curr_zip_size += feat_size
        zip_fh.write(s["audio"], arcname=feat_path.name)

        zip_info = zip_fh.getinfo(feat_path.name)
        offset = zip_info.header_offset + 30 + len(zip_info.filename)
        file_size = zip_info.file_size
        meta_fh.write(f"{s['id']}\t{curr_zip_path(curr_zip_idx)}:{offset}:{file_size}\n")

zip_fh.close()
