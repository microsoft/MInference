# Copyright (c) 2024 Microsoft
# Licensed under The MIT License [see LICENSE for details]

NUM_SAMPLES=80
REMOVE_NEWLINE_TAB=false
STOP_WORDS=""

if [ -z "${STOP_WORDS}" ]; then
    STOP_WORDS=""
else
    STOP_WORDS="--stop_words \"${STOP_WORDS}\""
fi

if [ "${REMOVE_NEWLINE_TAB}" = false ]; then
    REMOVE_NEWLINE_TAB=""
else
    REMOVE_NEWLINE_TAB="--remove_newline_tab"
fi

# task name in `synthetic.yaml`
synthetic=(
    "niah_single_1"
    "niah_single_2"
    "niah_single_3"
    "niah_multikey_1"
    "niah_multikey_2"
    "niah_multikey_3"
    "niah_multivalue"
    "niah_multiquery"
    "vt"
    "cwe"
    "fwe"
    "qa_1"
    "qa_2"
)
