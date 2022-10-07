#!/usr/bin/env python
import argparse
import os

from fairseq_cli.preprocess import main as preprocessing_main
from fairseq import tasks, options
from pathlib import Path
import sentencepiece as spm
from tqdm import tqdm
from itertools import product as cartesian
import gzip
import shutil

parser = argparse.ArgumentParser()
parser.add_argument('--data-path', default='/work/smt4/gao/tran/speech_translation/datasets/text/raw_data')
parser.add_argument('--out-path', default='/work/smt4/gao/tran/speech_translation/datasets/text/out')
parser.add_argument('--fairseq-task', default='translation')
parser.add_argument('--workers', type=int, default=8)
args = parser.parse_args()

DATA_PATH = Path(args.data_path)
DATA_PATH.mkdir(parents=True, exist_ok=True)

OUT_PATH = Path(args.out_path)
OUT_PATH.mkdir(parents=True, exist_ok=True)

# UPDATING GUIDE
# - If anything changed, data-bin should be removed
# - If train/valid/test split changed, also remove train.*/valid.*/test.*

# Assumed dictionary: DATA[dataset][{langs,lang1_path,lang2_path,etc...}]
# If not given, base_path defaults to '.' and langX_path defaults to langX.raw
# For each dataset, a 'base_path' can be specified.

DATA = {
    'mt.iwslt2020.src-asr': {
        'base_path': 'mt.iwslt2020.src-asr',
        'langs': ['en', 'de']
    },
    'mt.iwslt2020.src-asr.dev': {
        'base_path': 'mt.iwslt2020.src-asr.dev',
        'langs': ['en', 'de']
    },
    'mt.iwslt2020.src-asr.test2015': {
        'base_path': 'mt.iwslt2020.src-asr.test2015',
        'langs': ['en', 'de']
    },


    'train.mustc.src-normalized': {
            'base_path': 'train.mustc.src-normalized',
            'langs': ['en', 'de']
    },


    'parnia.iwslt2020_asr_en.transcripts': {
        'base_path': 'parnia.iwslt2020_asr_en.transcripts',
        'langs': ['en'],
    },
    'parnia.iwslt2020_mt_en-de.src-lowercase-nopunct': {
        'base_path': 'parnia.iwslt2020_mt_en-de.src-lowercase-nopunct',
        'langs': ['de', 'en'],
    },
    'parnia.iwslt2020_mt_en-de.raw': {
        'base_path': 'parnia.iwslt2020_mt_en-de.raw',
        'langs': ['de', 'en'],
    },
    'mustc-dev.src-unnormalized': {
        'base_path': 'dev.mustc',
        'langs': ['de', 'en'],
    },
    'mustc-dev.src-normalized': {
        'base_path': 'dev.mustc.src-normalized',
        'langs': ['de', 'en'],
    },
    'mustc-test-COMMON.src-unnormalized': {
        'base_path': 'test-COMMON.mustc',
        'langs': ['de', 'en'],
    },
    'mustc-test-COMMON.src-normalized': {
        'base_path': 'test-COMMON.mustc.src-normalized',
        'langs': ['de', 'en'],
    },
    'mustc-test-HE.src-unnormalized': {
        'base_path': 'test-HE.mustc',
        'langs': ['de', 'en'],
    },
    'mustc-test-HE.src-normalized': {
        'base_path': 'test-HE.mustc.src-normalized',
        'langs': ['de', 'en'],
    }
}

BPES = {
    # BPE on news_commentary_v15, separate for each language
    'parnia.iwslt2020_mt_en-de.32k.de':
        {'data': ['parnia.iwslt2020_mt_en-de.src-lowercase-nopunct'], 'langs': ['de'],
         'vocab_size': 32000, 'is_large': True},
    'parnia.iwslt2020_mt_en-de.5k.en':
        {'data': ['parnia.iwslt2020_mt_en-de.raw'], 'langs': ['de'],
         'vocab_size': 5000, 'is_large': True},

    'parnia.iwslt2020_mt_en-de.lowercase-nopunct.5k.en':
        {'data': ['parnia.iwslt2020_mt_en-de.src-lowercase-nopunct'], 'langs': ['en'],
         'vocab_size': 5000, 'is_large': True},
    'parnia.iwslt2020_mt_en-de.lowercase-nopunct.32k.en':
        {'data': ['parnia.iwslt2020_mt_en-de.src-lowercase-nopunct'], 'langs': ['en'],
         'vocab_size': 5000, 'is_large': True},
    'parnia.iwslt2020_asr_en.transcripts':
        {'data': ['parnia.iwslt2020_asr_en.transcripts'], 'langs': ['en'],
         'vocab_size': 5000, 'is_large': True},
    'parnia.iwslt2020.asr-mt-joint.en-5k':
        {'data': ['parnia.iwslt2020_asr_en.transcripts', 'parnia.iwslt2020_mt_en-de.src-lowercase-nopunct'],
         'langs': ['en'],
         'vocab_size': 5000, 'is_large': True},
    'parnia.iwslt2020.asr-mt-joint.all-5k':
        {'data': ['parnia.iwslt2020_asr_en.transcripts', 'parnia.iwslt2020_mt_en-de.src-lowercase-nopunct', 'parnia.iwslt2020_mt_en-de.src-lowercase-nopunct'],
         'langs': ['en', 'de'],
         'vocab_size': 5000, 'is_large': True},

    'mt.iwslt2020.src-asr.en':
        {'data': ['mt.iwslt2020.src-asr'], 'langs': ['en'],
         'vocab_size': 32000, 'is_large': True},
    'mt.iwslt2020.src-asr.de':
        {'data': ['mt.iwslt2020.src-asr'], 'langs': ['de'],
         'vocab_size': 32000, 'is_large': True}
}

GENERATE = {
    # IWSLT2020 32k-BPE
    'train.mt.iwslt2020.src-asr': {
        'type': 'train-dev',
        'train_data': ['mt.iwslt2020.src-asr'],
        'dev_data': ['mt.iwslt2020.src-asr.dev'],
        'directions': ['en-de'],
        'bpe': {'de': 'mt.iwslt2020.src-asr.de', 'en': 'mt.iwslt2020.src-asr.en'}
    },
    'eval.mt.iwslt2020.src-asr.test2015': {
        'type': 'test',
        'test_data': ['mt.iwslt2020.src-asr.test2015'],
        'directions': ['en-de'],
        'bpe': {'de': 'mt.iwslt2020.src-asr.de', 'en': 'mt.iwslt2020.src-asr.en'}
    },

    # MuST-C En-5k, De-32k
    'parnia.mustc-mt.en-de.en-5k.de-32k.src-normalized': {
        'type': 'train-dev',
        'train_data': ['parnia.iwslt2020_mt_en-de.src-lowercase-nopunct'],
        'dev_data': ['mustc-dev.src-normalized'],
        'directions': ['en-de'],
        'bpe': {'de': 'parnia.iwslt2020_mt_en-de.32k.de', 'en': 'parnia.iwslt2020_mt_en-de.lowercase-nopunct.5k.en'}
    },
    'parnia.mustc-mt.en-de.en-5k.de-32k.src-unnormalized': {
        'type': 'train-dev',
        'train_data': ['parnia.iwslt2020_mt_en-de.raw'],
        'dev_data': ['mustc-dev.src-unnormalized'],
        'directions': ['en-de'],
        'bpe': {'de': 'parnia.iwslt2020_mt_en-de.32k.de', 'en': 'parnia.iwslt2020_mt_en-de.5k.en'}
    },

    # MuST-C 32k
    'parnia.mustc-mt.en-de.32k.src-normalized.ft-mustc': {
        'type': 'train-dev',
        'train_data': ['train.mustc.src-normalized'],
        'dev_data': ['mustc-dev.src-normalized'],
        'directions': ['en-de'],
        'bpe': {'de': 'parnia.iwslt2020_mt_en-de.32k.de', 'en': 'parnia.iwslt2020_mt_en-de.lowercase-nopunct.32k.en'}
    },
    'parnia.mustc-mt.en-de.32k.src-normalized.fix': {
        'type': 'train-dev',
        'train_data': ['parnia.iwslt2020_mt_en-de.src-lowercase-nopunct'],
        'dev_data': ['mustc-dev.src-normalized'],
        'directions': ['en-de'],
        'bpe': {'de': 'parnia.iwslt2020_mt_en-de.32k.de', 'en': 'parnia.iwslt2020_mt_en-de.lowercase-nopunct.32k.en'}
    },
    'parnia.mustc-mt.en-de.en-5k.de-32k.joint.src-normalized.fix': {
        'type': 'train-dev',
        'train_data': ['parnia.iwslt2020_mt_en-de.src-lowercase-nopunct'],
        'dev_data': ['mustc-dev.src-normalized'],
        'directions': ['en-de'],
        'bpe': {'de': 'parnia.iwslt2020.asr-mt-joint.en-5k', 'en': 'parnia.iwslt2020_mt_en-de.lowercase-nopunct.32k.en'}
    },
    'parnia.mustc-mt.en-de.en-5k.de-32k.joint.src-normalized': {
        'type': 'train-dev',
        'train_data': ['parnia.iwslt2020_mt_en-de.src-lowercase-nopunct'],
        'dev_data': ['mustc-dev.src-normalized'],
        'directions': ['en-de'],
        'bpe': {'de': 'parnia.iwslt2020.asr-mt-joint.en-5k', 'en': 'parnia.iwslt2020_mt_en-de.lowercase-nopunct.32k.en'}
    },

    # MuST-C Joint 32k
    'parnia.mustc-mt.en-de.joint-5k.src-normalized': {
        'type': 'train-dev',
        'train_data': ['parnia.iwslt2020_mt_en-de.src-lowercase-nopunct'],
        'dev_data': ['mustc-dev.src-normalized'],
        'directions': ['en-de'],
        'bpe': {'de': 'parnia.iwslt2020.asr-mt-joint.all-5k', 'en': 'parnia.iwslt2020.asr-mt-joint.all-5k'}
    },
    'tstCOMMON.en-de.joint-5k.src-normalized': {
            'type': 'test',
            'test_data': ['mustc-test-COMMON.src-normalized'],
            'directions': ['en-de'],
            'bpe': {'de': 'parnia.iwslt2020.asr-mt-joint.all-5k', 'en': 'parnia.iwslt2020.asr-mt-joint.all-5k'}
        },
    'tstHE.en-de.joint-5k.src-normalized': {
        'type': 'test',
        'test_data': ['mustc-test-HE.src-normalized'],
        'directions': ['en-de'],
        'bpe': {'de': 'parnia.iwslt2020.asr-mt-joint.all-5k', 'en': 'parnia.iwslt2020.asr-mt-joint.all-5k'}
    },

    # Test sets
    'tstCOMMON.en-de.32k.src-normalized': {
        'type': 'test',
        'test_data': ['mustc-test-COMMON.src-normalized'],
        'directions': ['en-de'],
        'bpe': {'de': 'parnia.iwslt2020_mt_en-de.32k.de', 'en': 'parnia.iwslt2020_mt_en-de.lowercase-nopunct.32k.en'}
    },
    'tstHE.en-de.32k.src-normalized': {
        'type': 'test',
        'test_data': ['mustc-test-HE.src-normalized'],
        'directions': ['en-de'],
        'bpe': {'de': 'parnia.iwslt2020_mt_en-de.32k.de', 'en': 'parnia.iwslt2020_mt_en-de.lowercase-nopunct.32k.en'}
    },

    'mustc-mt.test-COMMON.en-de.en-5k.de-32k.src-normalized': {
        'type': 'test',
        'test_data': ['mustc-test-COMMON.src-normalized'],
        'directions': ['en-de'],
        'bpe': {'de': 'parnia.iwslt2020_mt_en-de.32k.de', 'en': 'parnia.iwslt2020_mt_en-de.lowercase-nopunct.5k.en'}
    },
    'mustc-mt.test-HE.en-de.en-5k.de-32k.src-normalized': {
        'type': 'test',
        'test_data': ['mustc-test-HE.src-normalized'],
        'directions': ['en-de'],
        'bpe': {'de': 'parnia.iwslt2020_mt_en-de.32k.de', 'en': 'parnia.iwslt2020_mt_en-de.lowercase-nopunct.5k.en'}
    },
    'mustc-mt.test-COMMON.en-de.en-5k.de-32k.src-unnormalized': {
        'type': 'test',
        'test_data': ['mustc-test-COMMON.src-unnormalized'],
        'directions': ['en-de'],
        'bpe': {'de': 'parnia.iwslt2020_mt_en-de.32k.de', 'en': 'parnia.iwslt2020_mt_en-de.5k.en'}
    },
    'mustc-mt.test-HE.en-de.en-5k.de-32k.src-unnormalized': {
        'type': 'test',
        'test_data': ['mustc-test-HE.src-unnormalized'],
        'directions': ['en-de'],
        'bpe': {'de': 'parnia.iwslt2020_mt_en-de.32k.de', 'en': 'parnia.iwslt2020_mt_en-de.5k.en'}
    },

}


def get_path(data, lang):
    path = [args.data_path]
    path += [DATA[data].get('base_path', '.')]
    path += [DATA[data].get(f"{lang}_path", f"{lang}.raw")]
    return os.path.join(*path)


def build_dictionary(filenames):
    task = tasks.get_task(args.fairseq_task)
    return task.build_dictionary(
        filenames,
        workers=args.workers,
    )


# STEP 1
def generate_bpe_vocab():
    bpe_tasks = [bpe for p in GENERATE.values() for bpe in p['bpe'].values()]

    out_path = OUT_PATH / 'bpe_vocab'
    out_path.mkdir(exist_ok=True, parents=True)
    (out_path / 'bpe_train').mkdir(exist_ok=True)

    spm_train = spm.SentencePieceTrainer.Train

    for bpe in bpe_tasks:
        datasets = BPES[bpe]['data']
        langs = BPES[bpe]['langs']

        inputs = []
        for lang in langs:
            for dataset in datasets:
                if lang in DATA[dataset]['langs']:
                    inputs += [get_path(dataset, lang)]

        print(f' * Training {bpe} ...')
        if (out_path / f"{bpe}.model").exists():
            print(f'  ⬤  Already exists, nothing to do.')
        else:
            vocab_size = BPES[bpe].get('vocab_size', 32000)
            sample_k = BPES[bpe].get('sample_k', -1)
            print(f"  + Using vocab size {vocab_size} and " + (
                'the whole dataset' if sample_k > 0 else f"{sample_k} samples from the dataset."))
            spm_train(
                input=','.join(inputs),
                model_prefix=str(out_path / bpe),
                model_type='bpe',
                vocab_size=vocab_size,
                shuffle_input_sentence=True,
                input_sentence_size=sample_k,
                train_extremely_large_corpus=BPES[bpe].get('is_large', False)
            )

        print(f' * BPE encode training files {bpe} ...')
        paths_bpe_corpuses = []
        for lang in langs:
            for dataset in datasets:
                if lang in DATA[dataset]['langs']:
                    in_file_path = get_path(dataset, lang)
                    sp = spm.SentencePieceProcessor(model_file=f'{out_path / bpe}.model')
                    print(f' * Encoding {in_file_path} ...')

                    raw_target_path = out_path / 'bpe_train' / f'{bpe}.{dataset}.raw.{lang}.gz'
                    target_path = out_path / 'bpe_train' / f'{bpe}.{dataset}.enc.{lang}'

                    paths_bpe_corpuses += [str(target_path)]

                    if not os.path.isfile(target_path):
                        with gzip.open(raw_target_path, 'wt') as raw_out_file, open(target_path, 'wt') as out_file:
                            with open(in_file_path, 'rt') as in_file:
                                for in_line in tqdm(in_file):
                                    out_line = sp.Encode(in_line, out_type='str')
                                    out_file.write(' '.join(out_line))
                                    out_file.write('\n')
                                    raw_out_file.write(in_line)
                        print(f' * Done encoding {lang}@{dataset}.')
                    else:
                        print(f'  ⬤  Already encoded, skipping (delete {target_path} to update) ...')

        print(f" * Generating dictionary on {bpe} ...")
        if (out_path / f"{bpe}.fairseq_vocab").exists():
            print(f'  ⬤  Already exists, nothing to do.')
        else:
            build_dictionary(paths_bpe_corpuses).save(str(out_path / f"{bpe}.fairseq_vocab"))


# STEP 2
def bpe_encode_data():
    for gen_name, gen in GENERATE.items():
        out_path = OUT_PATH / gen_name
        out_path.mkdir(exist_ok=True)

        for lang, bpe in gen['bpe'].items():
            sp = spm.SentencePieceProcessor(model_file=f'{OUT_PATH / "bpe_vocab" / bpe}.model')
            subsets = ['train', 'dev'] if gen['type'] == 'train-dev' else ['test']
            for subset in subsets:
                data_dict = gen[f'{subset}_data']
                if isinstance(data_dict, list):
                    data_dict = {pair: data_dict for pair in gen['directions']}

                for pair, datasets in data_dict.items():
                    src_lang, tgt_lang = pair.split('-')
                    rev_pair = f"{tgt_lang}-{src_lang}"
                    if lang not in [src_lang, tgt_lang]:
                        continue

                    print(f' * Encoding direction {pair} on {subset}.{lang}@{gen_name}')

                    raw_target_path = out_path / f'{subset}.raw.{pair}.{lang}.gz'
                    target_path = out_path / f'{subset}.enc.{pair}.{lang}'

                    rev_raw_target_path = out_path / f'{subset}.raw.{rev_pair}.{lang}.gz'
                    rev_target_path = out_path / f'{subset}.enc.{rev_pair}.{lang}'

                    # Reverse direction
                    if not src_lang == tgt_lang:
                        if not rev_raw_target_path.is_symlink() and not rev_raw_target_path.is_file():
                            rev_raw_target_path.symlink_to(raw_target_path)
                        if not rev_target_path.is_symlink() and not rev_target_path.is_file():
                            rev_target_path.symlink_to(target_path)

                    if not os.path.isfile(target_path):
                        with gzip.open(raw_target_path, 'wt') as raw_out_file, open(target_path, 'wt') as out_file:
                            for dataset in datasets:
                                raw_cached_path = OUT_PATH / 'bpe_vocab' / 'bpe_train' / f'{bpe}.{dataset}.raw.{lang}.gz'
                                cached_path = OUT_PATH / 'bpe_vocab' / 'bpe_train' / f'{bpe}.{dataset}.enc.{lang}'

                                if cached_path.exists():
                                    print(
                                        f'  ⬤  Cached version found from BPE generation (delete {cached_path} to update) ...')
                                    with cached_path.open('rt') as cached_file:
                                        shutil.copyfileobj(cached_file, out_file)
                                    with gzip.open(raw_cached_path, 'rt') as raw_cached_file:
                                        shutil.copyfileobj(raw_cached_file, raw_out_file)
                                else:
                                    p = get_path(dataset, lang)
                                    if not os.path.exists(p):
                                        print(
                                            f'  ⚠  No data for {lang} on subset {subset} for dataset {dataset}, skipping ...')
                                    else:
                                        with open(p, 'rt') as in_file:
                                            for in_line in tqdm(in_file):
                                                if in_line.strip().upper() == '<UNK>':
                                                    out_file.write("<unk>\n")
                                                else:
                                                    out_line = sp.Encode(in_line, out_type='str')
                                                    out_file.write(' '.join(out_line))
                                                    out_file.write('\n')
                                                raw_out_file.write(in_line)
                                        print(f' * Done encoding {subset}.{lang}@{gen_name}')
                    else:
                        print(f'  ⬤  Already encoded, skipping (delete {target_path} to update) ...')


def fairseq_binarization():
    def binarize(src_lang, tgt_lang, trainpref, validpref, destdir, src_dict, tgt_dict, workers):
        __parser = options.get_preprocessing_parser()

        __args = __parser.parse_args(args=['--source-lang', str(src_lang),
                                           '--target-lang', str(tgt_lang),
                                           '--srcdict', str(src_dict),
                                           '--tgtdict', str(tgt_dict),
                                           '--trainpref', str(trainpref),
                                           '--validpref', str(validpref),
                                           '--destdir', str(destdir),
                                           '--workers', str(workers)])
        preprocessing_main(__args)

    for gen_name, gen in GENERATE.items():
        out_path = OUT_PATH / gen_name / 'data-bin'
        out_path.mkdir(exist_ok=True)

        if gen['type'] == 'test':
            print(f" * Not binarizing test dataset {gen_name}.")
            continue

        for direction in gen['directions']:
            print(f' * Binarizing {direction}@{gen_name} ...')
            src_lang, tgt_lang = direction.split('-')
            src_bpe, tgt_bpe = gen['bpe'][src_lang], gen['bpe'][tgt_lang]
            enc_path = OUT_PATH / gen_name

            # check existence
            if all((enc_path / 'data-bin' / f'{_split}.{direction}.{_lang}.bin').exists()
                   for _split, _lang in cartesian(['train', 'valid'], [src_lang, tgt_lang])):
                print(f'  ⬤  Already binarized, skipping (delete {enc_path / "data-bin"} to update) ...')
                continue

            binarize(
                src_lang, tgt_lang,
                trainpref=enc_path / f'train.enc.{direction}',
                validpref=enc_path / f'dev.enc.{direction}',
                destdir=out_path,
                src_dict=OUT_PATH / 'bpe_vocab' / f'{src_bpe}.fairseq_vocab',
                tgt_dict=OUT_PATH / 'bpe_vocab' / f'{tgt_bpe}.fairseq_vocab',
                workers=args.workers
            )


if __name__ == '__main__':
    print("==================================")
    print('STEP 1: Train BPE on training sets')
    print("==================================")
    generate_bpe_vocab()

    print("============================================================")
    print('STEP 2: BPE-Encode data, generate fairseq subword vocabulary')
    print("===========================================================")
    bpe_encode_data()

    print("=================================")
    print('STEP 3: Binarize data for fairseq')
    print("=================================")
    fairseq_binarization()

    print('All done.')
