# d23a-fusion: Analysis code for "Fusion of Probabilistic Projections of Sea Level"

[![DOI](https://zenodo.org/badge/630738591.svg)](https://zenodo.org/badge/latestdoi/630738591)

## Usage guidelines
This repository accompanies the following manuscript:

B. S. Grandey, J. Dauwels, B. P. Horton, and L. Y. Chew (2023),  **Fusion of Probabilistic Projections of Sea Level**, Preprint at _Research Square_, [doi:10.21203/rs.3.rs-2922142/v1](https://doi.org/10.21203/rs.3.rs-2922142/v1).

The manuscript serves as the primary reference.
The Zenodo archive of this repository serves as a secondary reference.

## Workflow

### Environment
To create a _conda_ environment with the necessary software dependencies, use the [**`environment.yml`**](environment.yml) file:

```
conda env create --file environment.yml
conda activate d23a-fusion
```

The analysis has been performed within this environment on _macOS 13_ (arm64).

### Input data
The analysis code requires data from the [IPCC AR6 Sea Level Projections](https://doi.org/10.5281/zenodo.6382554) and the [IPCC AR6 Relative Sea Level Projection Distributions](https://doi.org/10.5281/zenodo.5914932) repositories, which can be downloaded as follows:

```
mkdir data
curl "https://zenodo.org/record/6382554/files/ar6.zip?download=1" --output data/ar6.zip
unzip data/ar6.zip -d data/
curl "https://zenodo.org/record/6382554/files/ar6-regional-confidence.zip?download=1" --output data/ar6-regional-confidence.zip
unzip data/ar6-regional-confidence.zip -d data/
curl "https://zenodo.org/record/5914932/files/ar6-regional-distributions.zip?download=1" --output data/ar6-regional-distributions.zip
unzip data/ar6-regional-distributions.zip -d data/
```

Users of these data should note the [required acknowledgments and citations](https://doi.org/10.5281/zenodo.6382554).

### Analysis
Analysis is performed using [**`fusion_analysis_d23a.ipynb`**](fusion_analysis_d23a.ipynb), which writes figures to [**`figs_d23a/`**](figs_d23a)

## Author
[Benjamin S. Grandey](https://grandey.github.io) (_Nanyang Technological University_), in collaboration with Justin Dauwels, Benjamin P. Horton, and Lock Yue Chew.

## Acknowledgements
This Research/Project is supported by the National Research Foundation, Singapore, and National Environment Agency, Singapore under the National Sea Level Programme Funding Initiative (Award No. USS-IF-2020-3).
We thank the projection authors for developing and making the sea level rise projections available, multiple funding agencies for supporting the development of the projections, and the NASA Sea Level Change Team for developing and hosting the IPCC AR6 Sea Level Projection Tool.
