# Copyright (c) 2025 Microsoft
# Licensed under The MIT License [see LICENSE for details]

pushd $(dirname "$0") > /dev/null

  pip install transformers

  pip install jieba rouge
  pip install tree-sitter==0.21.3

  pip install datasets jsonlines fire matplotlib pandas seaborn tqdm
  pip install accelerate
  pip install jieba mysql-connector-python fuzzywuzzy rouge jsonlines SentencePiece
  pip install git+https://github.com/NVIDIA/NeMo.git
  pip install nltk  hydra-core wonderwords lightning lhotse jiwer librosa pyannote-core webdataset editdistance pyannote.metrics tenacity xopen
  pip install html2text bs4
  python -c 'import nltk; nltk.download("punkt_tab")'

  cd ruler/data/synthetic/json
  python download_paulgraham_essay.py
  bash download_qa_dataset.sh

popd > /dev/null
