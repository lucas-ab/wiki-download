#!/usr/bin/env bash
# Script to download a Wikipedia dump
# and preprocess it to ids

# Script is partially based on https://github.com/facebookresearch/fastText/blob/master/get-wikimedia.sh
ROOT="../data"
DUMP_DIR="${ROOT}/wiki_dumps"
EXTR_DIR="${ROOT}/wiki_extr"
WIKI_DIR="${ROOT}/wiki"
EXTR="wikiextractor"
mkdir -p "${ROOT}"
mkdir -p "${DUMP_DIR}"
mkdir -p "${EXTR_DIR}"
mkdir -p "${WIKI_DIR}"

echo "Saving data in ""$ROOT"
read -r -p "Choose a language (e.g. en, bh, fr, etc.): " choice
LANG="$choice"
echo "Chosen language: ""$LANG"
DUMP_FILE="${LANG}wiki-latest-pages-articles.xml.bz2"
DUMP_PATH="${DUMP_DIR}/${DUMP_FILE}"

if [ ! -f "${DUMP_PATH}" ]; then
  read -r -p "Continue to download (WARNING: This might be big and can take a long time!) (y/n)? " choice
  case "$choice" in
    y|Y ) echo "Starting download...";;
    n|N ) echo "Exiting";exit 1;;
    * ) echo "Invalid answer";exit 1;;
  esac
  wget -c "https://dumps.wikimedia.org/""${LANG}""wiki/latest/""${DUMP_FILE}""" -P "${DUMP_DIR}"
else
  echo "${DUMP_PATH} already exists. Skipping download."
fi

# Check if directory exists
if [ ! -d "${EXTR}" ]; then
  git clone https://github.com/attardi/wikiextractor.git
  cd "${EXTR}"
  python setup.py install
fi

EXTR_PATH="${EXTR_DIR}/${LANG}"
if [ ! -d "${EXTR_PATH}" ]; then
  read -r -p "Continue to extract Wikipedia (WARNING: This might take a long time!) (y/n)? " choice
  case "$choice" in
    y|Y ) echo "Extracting ${DUMP_PATH} to ${EXTR_PATH}...";;
    n|N ) echo "Exiting";exit 1;;
    * ) echo "Invalid answer";exit 1;;
  esac
  python wikiextractor/WikiExtractor.py -s --json -o "${EXTR_PATH}" "${DUMP_PATH}"
else
  echo "${EXTR_PATH} already exists. Skipping extraction."
fi

OUT_PATH="${WIKI_DIR}/${LANG}"
if [ ! -f "${OUT_PATH}/val.csv" ]; then
  read -r -p "Continue to merge Wikipedia articles (y/n)? " choice
  case "$choice" in
    y|Y ) echo "Merging articles from ${EXTR_PATH} to ${OUT_PATH}...";;
    n|N ) echo "Exiting";exit 1;;
    * ) echo "Invalid answer";exit 1;;
  esac
  python merge_wiki.py -i "${EXTR_PATH}" -o "${OUT_PATH}"
else
  echo "${OUT_PATH}/val.csv already exists. Skipping merging."
fi

python -m spacy download pt

NP_PATH="${OUT_PATH}"
read -r -p "Continue to tokenize the corpus (y/n)? " choice
case "$choice" in
    y|Y ) echo "Tokening corpus to ${NP_PATH}...";;
    n|N ) echo "Exiting";exit 1;;
    * ) echo "Invalid answer";exit 1;;
  esac
  python create_toks.py --dir-path "${NP_PATH}" --chunksize 50000

# T2I_PATH="${NP_PATH}/tmp"
# if [ ! -d "${T2I_PATH}" ]; then
# 	mkdir "${T2I_PATH}"
# fi


read -r -p "Continue to numericalize the corpus (y/n)?" choice
case $choice in
    y|Y ) echo "Numericalizing the corpus to ${OUT_PATH}...";;
    n|N ) echo "Exiting";exit 1;;
    * ) echo "Invalid answer";exit 1;;
  esac
  python merge_np_files.py --dir-path "${T2I_PATH}" --max-vocab 100000
