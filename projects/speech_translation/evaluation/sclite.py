import argparse
import re
import shlex
from operator import itemgetter
import os
import tempfile

def create_ctm(segments, glm, os_path):
    tf1 = tempfile.NamedTemporaryFile('wt', suffix='.ctm')

    tf1.write(";; <name> <track> <start> <duration> <word> <confidence> [<n-best>]\n")
    start = 0.0  # we invent some dummy start offsets and durations
    for idx, hyp_line in enumerate(hyps):
        tf1.write(f";; {segments[idx]} ({start + 0.01}-{start + 0.99})\n")
        if hyp_line:
            words = hyp_line.split()
            word_duration = 0.9 / len(words)

            for i in range(len(words)):
                if '^;' in words[i] or '[UNK]' in words[i]:
                    continue
                tf1.write(f"{segments[idx]} 1 {start + 0.01 + i * word_duration} {word_duration} {words[i]}\n")
        start += 1
    tf1.flush()

    tf2 = tempfile.NamedTemporaryFile('wt', suffix='.ctm')

    os.system(f"csrfilt.sh -i ctm -t hyp -s {glm} < {tf1.name} | awk '$5 !~ /[0-9]\./ {{print $0}}' > {tf2.name}")

    tf1.close()
    return tf2


def post_process_hyp(txt):
    txt = txt.replace(" 0 ", " zero ").replace(" 10 ", " zero ")
    txt = re.sub(r" OKS\b", " okay", txt)
    return txt


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('ref')
    parser.add_argument('hyp')
    parser.add_argument('--stm_path', default='/u/vtran/eval/asr')
    parser.add_argument('--sctk-path', default="/work/speech/tools/sctk-2.4.0/bin")
    parser.add_argument('--glm', default='/u/vtran/eval/asr/english.glm')
    args = parser.parse_args()

    assert os.path.isdir(args.sctk_path)
    os.environ["PATH"] = os.path.realpath(args.sctk_path) + ":" + os.environ["PATH"]

    stm_path = os.path.join(args.stm_path, args.ref + ".stm.filt")

    with open(stm_path, 'rt') as f:
        segments = list(e.split()[0] for e in f.readlines())

    with open(args.hyp, 'rt') as f:
        hyps = list(filter(lambda s: s.startswith('H-'), f.readlines()))
        hyps = list(re.sub(r'^H-', '', s) for s in hyps)
        hyps = list(s.split('\t') for s in hyps)
        hyps = list((int(idx), text.replace(" ", "").replace("▁", " ").strip()) for idx, _, text in hyps)
        hyps = list(sorted(hyps, key=itemgetter(0)))
        hyps = list(post_process_hyp(text) for _, text in hyps)
        hyps = list(hyps)

    ctm_file = create_ctm(segments, args.glm, None)

    output_path = os.path.dirname(os.path.realpath(args.hyp))

    sclite_args = [
        "sclite",
        "-r", stm_path, "stm",
        "-h", ctm_file.name, "ctm",
        "-o", "all",
        "-o", "dtl",
        "-n", "sclite",
        "-O", os.path.dirname(os.path.realpath(args.hyp))
    ]

    os.system(shlex.join(sclite_args))

    ctm_file.close()

    wer = -1
    with open(os.path.join(output_path, 'sclite.dtl')) as f:
        for line in f:
            if line.startswith("Percent Total Error"):
                errors = float("".join(line.split()[5:])[1:-1])
                print(f'Errors: {errors}')
            if line.startswith("Ref. words"):
                wer = 100.0 * errors / float("".join(line.split()[3:])[1:-1])
                break

    print(f"WER: {wer}")
