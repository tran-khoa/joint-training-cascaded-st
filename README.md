# Does Joint Training Really Help Cascaded Speech Translation?

This repository contains code for the paper "Does Joint Training Really Help Cascaded Speech Translation?" in [EMNLP 2022](https://2022.emnlp.org/),
based on [fairseq](https://github.com/facebookresearch/fairseq).

## Cite This Work
To cite this work, please use the following .bib:
```
@InProceedings{tran22:uniblock,
	author={Tran, Viet Anh Khoa and Thulke, David and Gao, Yingbo and Herold, Christian and Ney, Hermann},  	
	title={Does Joint Training Really Help Cascaded Speech Translation?},  
	booktitle={Conference on Empirical Methods in Natural Language Processing},
	year=2022,  
	address={Abu Dhabi, United Arab Emirates},  
	month=nov,  
	booktitlelink={https://2022.emnlp.org/},
}
```

## Requirements and Installation (adapted from fairseq)
* [PyTorch](http://pytorch.org/) version 1.7.1
* torchaudio 0.7.2
* Python version >= 3.7
* **To install fairseq** and develop locally:

``` bash
git clone https://github.com/tran-khoa/joint-training-cascaded-st
cd joint-training-cascaded-st
pip install --editable ./
cd projects/speech_translation
pip install -r requirements.txt

# on MacOS:
# CFLAGS="-stdlib=libc++" pip install --editable ./
```

* **For faster training** install NVIDIA's [apex](https://github.com/NVIDIA/apex) library:

``` bash
git clone https://github.com/NVIDIA/apex
cd apex
pip install -v --no-cache-dir --global-option="--cpp_ext" --global-option="--cuda_ext" \
  --global-option="--deprecated_fused_adam" --global-option="--xentropy" \
  --global-option="--fast_multihead_attn" ./
```

# Running experiments
The implementation is located in `projects/speech_translation`. 
Please refer to the scripts in `projects/speech_translation/experiments`.
The term `joint-seq` refers to `Top-K-Train` in the paper, `tight` refers to 'Tight-Integration' as introduced in [Tight integrated end-to-end training for cascaded speech translation](https://ieeexplore.ieee.org/abstract/document/9383462).

# License (adapted from fairseq)

fairseq(-py) is MIT-licensed.
The license applies to the pre-trained models as well.

