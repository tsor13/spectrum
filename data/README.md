# Data Directory

This README has instructions for hydrating the Spectrum Suite dataset.

If you utilize any particular dataset in your research, please cite the corresponding data source. Citations for each source can be found in Appendix A of our [preprint](https://arxiv.org/pdf/2510.06084).

Note that while all data needs to be downloaded for full replicability, we believe the results should generalize even when including just a subset of the data. Additionally, we encourage people to extend our unified format to include more datasets!

## Automatic Downloads

To download as much data as possible automatically, run:

```bash
bash download.sh
```

The script will automatically download all datasets that can be obtained via command line tools (git, wget, curl). However, some datasets require manual intervention due to authentication requirements, website restrictions, or author permissions.

## Manual Downloads Required

After running the download script, you will need to manually download the following datasets:

### Numbergame
- **Sources**:
  - https://openpsychologydata.metajnl.com/articles/10.5334/jopd.19
  - https://dataverse.harvard.edu/dataset.xhtml?persistentId=doi:10.7910/DVN/A8ZWLF
- **Directory**: `numbergame/`

### OpinionQA
- **Source**: https://worksheets.codalab.org/bundles/0x050b7e72abb04d1f9b493c1743e580cf
- **Directory**: `opinionqa/`
- **Instructions**: After download, extract with `tar -xvf human_resp.tar.gz`

### WVS (World Values Survey)
- **Source**: https://www.worldvaluessurvey.org/WVSDocumentationWV7.jsp
- **Files to download**:
  - WVS Cross-National Wave 7 csv v6 0.zip
  - WVS Cross-National Inverted Wave 7 sav v6 0.zip
- **Directory**: `wvs/`
- **Instructions**: Unzip files in the wvs directory

### Netflix
- **Source**: https://www.kaggle.com/api/v1/datasets/download/netflix-inc/netflix-prize-data
- **Directory**: `netflix/`
- **Requirements**: May require Kaggle API authentication

### MPI
- **Source**: https://drive.google.com/file/d/1MOE4y_nGJiYU_vxCqnWSiYIKCk-dqPJE/view?usp=sharing
- **Directory**: `mpi/`

### Rotten Tomatoes
- **Source**: https://drive.google.com/file/d/12IpMErb4j83h5gGTdTpv0WZOf5ceY7b3/view
- **Directory**: `rotten_tomatoes/`

### Jeopardy
- **Source**: https://drive.google.com/file/d/0BwT5wj_P7BKXUl9tOUJWYzVvUjA/view?resourcekey=0-uFrn8bQkUfSCvJlmtKGCdQ
- **Directory**: `misc/`

## Request from authors
In order to download the following datasets, you will need to request them from the corresponding paper authors.

### Hatespeech (Kumar)
- **Status**: Request from Deepak Kumar, with a description of the intended use case, as described [here](https://data.esrg.stanford.edu/study/toxicity-perspectives). Note: This is not the file downloadable from the link, as it does not contain annotator IDs - instead, it requires worker hashes (must request from authors).
- **Directory**: `hatespeech/`
- **Target file**: `classified_data_final_w_worker_hash.json`

## Automatically Downloaded Datasets

The following datasets will be downloaded automatically by the script:

- **Polis**: GitHub repository from compdemocracy/openData
- **Diffuse distributions**: GitHub repository from y0mingzhang/diffuse-distributions
- **Habermas**: Direct downloads from Google Cloud Storage
- **Generative Social Choice**: GitHub repository (renamed to generativesocialchoice)
- **Popquorn**: GitHub repository from Jiaxin-Pei/Potato-Prolific-Dataset
- **Change my view**: Direct download from Zenodo (may take a while due to file size)
- **VITAL**: GitHub repository from anudeex/VITAL
- **Chemistry**: Direct downloads from GitHub raw files
- **NYTimes Books**: GitHub repository from nicolemeister/benchmarking-distributional-alignment
- **GPQA**: GitHub repository from idavidrein/gpqa

The remaining datasets are either automatically loaded from e.g. Hugging Face, or are synthetically generated.