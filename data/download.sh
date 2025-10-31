#!/bin/bash

set -e  # Exit on any error

echo "Starting data download script..."

### BASH DOWNLOADS

# Polis
echo "Downloading Polis data..."
git clone git@github.com:compdemocracy/openData.git

# Diffuse distributions
echo "Downloading Diffuse distributions..."
git clone git@github.com:y0mingzhang/diffuse-distributions.git

# DICES
echo "Downloading DICES data..."
git clone git@github.com:google-research-datasets/dices-dataset.git

# Habermas
echo "Setting up Habermas data..."
mkdir -p habermas_data
cd habermas_data
wget https://storage.googleapis.com/habermas_machine/datasets/hm_all_candidate_comparisons.parquet
wget https://storage.googleapis.com/habermas_machine/datasets/hm_all_final_preference_rankings.parquet
wget https://storage.googleapis.com/habermas_machine/datasets/hm_all_position_statement_ratings.parquet
wget https://storage.googleapis.com/habermas_machine/datasets/hm_all_round_survey_responses.parquet
cd ..

# Baby names
echo "Downloading Baby names data..."
mkdir -p baby_names
cd baby_names
wget https://www.ssa.gov/oact/babynames/names.zip
unzip names.zip
rm names.zip
cd ..

# Haiku RNN
echo "Downloading Haiku RNN data..."
git clone git@github.com:docmarionum1/haikurnn.git

# Ambient
echo "Downloading Ambient data..."
git clone git@github.com:alisawuffles/ambient.git

### MANUAL DOWNLOADS

# Hatespeech (Kumar)
echo "Creating Hatespeech directory..."
mkdir -p hatespeech
# NOTE: Hatespeech data requires manual request from author

# Numbergame
echo "Creating Numbergame directory..."
mkdir -p numbergame
# NOTE: Manual download required from:
# - https://openpsychologydata.metajnl.com/articles/10.5334/jopd.19
# - https://dataverse.harvard.edu/dataset.xhtml?persistentId=doi:10.7910/DVN/A8ZWLF

# OpinionQA
echo "Creating OpinionQA directory..."
mkdir -p opinionqa
cd opinionqa
# NOTE: Manual download required from: https://worksheets.codalab.org/bundles/0x050b7e72abb04d1f9b493c1743e580cf
# After download: tar -xvf human_resp.tar.gz
cd ..

# WVS (World Values Survey)
echo "Creating WVS directory..."
mkdir -p wvs
# NOTE: Manual download required:
# Download 'WVS Cross-National Wave 7 csv v6 0.zip' from https://www.worldvaluessurvey.org/WVSDocumentationWV7.jsp
# and 'WVS Cross-National Inverted Wave 7 sav v6 0.zip'
# Then unzip in wvs directory

# Generative Social Choice
echo "Downloading Generative Social Choice data..."
git clone git@github.com:generative-social-choice/survey_data.git
mv survey_data generativesocialchoice

# Popquorn
echo "Downloading Popquorn data..."
git clone git@github.com:Jiaxin-Pei/Potato-Prolific-Dataset.git

# Change my view
echo "Downloading Change my view data..."
URL="https://zenodo.org/records/3778298/files/threads.jsonl.bz2?download=1"
DEST_DIR="changemyview"
DEST_FILE="${DEST_DIR}/threads.jsonl"
TMP_FILE="$(mktemp --suffix=.bz2)"

mkdir -p "${DEST_DIR}"
curl -L "${URL}" -o "${TMP_FILE}"
bzip2 -dc "${TMP_FILE}" > "${DEST_FILE}"
rm -f "${TMP_FILE}"

# Netflix
echo "Downloading Netflix data..."
NETFLIX_URL="https://www.kaggle.com/api/v1/datasets/download/netflix-inc/netflix-prize-data"
NETFLIX_DIR="netflix"

mkdir -p "${NETFLIX_DIR}"
# NOTE: Netflix download may require Kaggle API authentication
# curl -L "${NETFLIX_URL}" -o "${NETFLIX_DIR}/netflix-prize-data.zip"
# cd netflix && unzip netflix-prize-data.zip && rm -f netflix-prize-data.zip && cd ..

# NOTE: After Netflix data download, run these Python scripts:
# python random_classes/dataset_loaders/netflix_compute_avg_ratings.py \
#     --data-dir data/netflix \
#     --combined-pattern "combined_data_*.txt" \
#     --movies-csv movie_titles.csv \
#     --min-votes 1000 \
#     --output data/netflix/avg_ratings.jsonl
#
# python random_classes/dataset_loaders/netflix_users_by_similarity.py \
#   --data-dir data/netflix \
#   --min-ratings 500 \
#   --output data/netflix/users.jsonl

# VITAL Distributional Evaluation Datasets
echo "Downloading VITAL datasets..."
git clone git@github.com:anudeex/VITAL.git
# This provides:
# - VITAL/dataset/vital_distributional_moralchoice.json - 181 moral scenarios with human distribution labels
# - VITAL/dataset/vital_distributional_globalopinionqa.json - 1676 global opinion questions with demographic-specific distributions

# Chemistry
echo "Downloading Chemistry datasets..."
mkdir -p chemistry
cd chemistry
wget https://raw.githubusercontent.com/ur-whitelab/BO-ICL/refs/heads/main/paper/dataset/oxidative_methane_coupling.csv
wget https://raw.githubusercontent.com/ur-whitelab/BO-ICL/refs/heads/main/paper/dataset/data/esol_iupac.csv
# train_data_num_feats.csv
wget https://raw.githubusercontent.com/ur-whitelab/BO-ICL/refs/heads/main/paper/dataset/data/train_data_num_feats.csv
cd ..

# NYTimes Books
echo "Downloading NYTimes Books data..."
git clone git@github.com:nicolemeister/benchmarking-distributional-alignment.git

# MPI
echo "Creating MPI directory..."
mkdir -p mpi
git clone git@github.com:jfisher52/AI_Pluralistic_Alignment.git
# NOTE: Additional manual download required from: https://drive.google.com/file/d/1MOE4y_nGJiYU_vxCqnWSiYIKCk-dqPJE/view?usp=sharing

# BARE
echo "Downloading BARE data..."
git clone git@github.com:pgasawa/BARE.git

# Rotten Tomatoes
echo "Creating Rotten Tomatoes directory..."
mkdir -p rotten_tomatoes
# NOTE: Manual download required from: https://drive.google.com/file/d/12IpMErb4j83h5gGTdTpv0WZOf5ceY7b3/view

# GPQA
echo "Downloading GPQA data..."
git clone git@github.com:idavidrein/gpqa.git
cd gpqa
unzip -P deserted-untie-orchid dataset.zip
cd ..

# mkdir diverse_valid_data
mkdir -p diverse_valid_data

# car models
cd diverse_valid_data
echo "Downloading car models data..."
git clone git@github.com:abhionlyone/us-car-models-data.git
cd -


echo "Data download script completed!"
echo ""
echo "Some datasets require manual download. See README.md for detailed instructions."



# # BARE
# echo "Creating BARE directory..."
# mkdir -p BARE
# # NOTE: BARE data requires manual download from authors

# LeWiDi
echo "Creating LeWiDi directory..."
mkdir -p lewidi
# NOTE: LeWiDi data requires manual download from: https://www.codabench.org/competitions/7192/#/pages-tab