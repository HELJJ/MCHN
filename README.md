# Multi-Channel Hypergraph-Aware-Healthcare

Codes for  paper: Multi-Channel Hypergraph Network for Sequential Diagnosis Prediction in Healthcare

## Download the MIMIC-III and MIMIC-IV datasets
Go to [https://mimic.physionet.org/](https://mimic.physionet.org/gettingstarted/access/) for access. Once you have the authority for the dataset, download the dataset and extract the csv files to `data/mimic3/raw/` and `data/mimic4/raw/` in this project.

## Preprocess

For the MIMIC-III dataset: 

```bash
python run_preprocess_mimic-iii.py
```

For the MIMIC-IV dataset: 

```
python run_preprocess_mimic-iv.py
```

## Train model

```bash
python train_hyper.py
```

## Configuration
Please see `train_hyper.py` for detailed configurations.

## Enviroment

Here we list our runing environment.

```
python==3.6.13
torch==1.10.1
tqdm==4.62.3
scipy==1.5.4
scikit-learn==1.3.1
numpy==1.24.3
torch-geometric==2.2.0
torch-scatter==2.0.9
torch-sparse==0.6.12
```

You can also use requirements.txt to install the environment using the command pip install requirements.txt -i https://pypi.mirrors.ustc.edu.cn/simple

## Folder Specification

- data/
  - mimic3
    - encoded
    - parsed
    - raw
    - standard
  - mimic4
    - encoded
    - parsed
    - raw
    - standard
  - params
    - mimic3
    - mimic4
  - icd9.txt
- raw
  - mimic3
    - ADMISSIONS.csv
    - DIAGNOSES_ICD.csv
    - PRESCRIPTIONS.csv
  - mimic4
    - admissions.csv
    - diagnoses_ics.csv
    - prescriptions.csv

## How to run

```
python train_hyper.py
```
