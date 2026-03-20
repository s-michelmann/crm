# Data Directory

This directory should contain the datasets needed to run the analyses.
Data files are not included in the repository due to size constraints.

## Expected Structure

```
data/
├── Podcast/          # Podcast ECoG dataset (BIDS format, 9 subjects)
├── THINGSbetas/      # THINGS fMRI response data
└── reagh/            # Reagh et al. fMRI data
```

## How to Obtain the Data

### Podcast ECoG
- Zada et al. (2025), available from the original authors
- BIDS-formatted iEEG recordings from 9 subjects

### THINGS fMRI
- THINGS fMRI beta images (Hebart et al.)
- Download preprocessed betas and place in `THINGSbetas/`

### Reagh et al. fMRI
- Reagh et al. (2023) memory reinstatement dataset
- Place in `reagh/`

## Setup

After downloading, either place datasets directly in this directory or create symlinks:

```bash
ln -s /path/to/your/Podcast data/Podcast
ln -s /path/to/your/THINGSbetas data/THINGSbetas
ln -s /path/to/your/reagh data/reagh
```
