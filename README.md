# Multi-Channel Hypergraph Network for Sequential Diagnosis Prediction in Healthcare

This is the code repository for the paper "Multi-Channel Hypergraph Network for Sequential Diagnosis Prediction in Healthcare". 

## Requirements

python==3.6.13
torch==1.10.1
tqdm==4.62.3
scipy==1.5.4
scikit-learn==1.3.1
numpy==1.24.3
torch-geometric==2.2.0
torch-scatter==2.0.9
torch-sparse==0.6.12

## 1. Prepare the datasets

### (1) Download the MIMIC-III and MIMIC-IV datasets

The original datasets can be downloaded from the following URLs.

Go to [https://mimic.physionet.org/](https://mimic.physionet.org/gettingstarted/access/) for access. Once you have the authority for the dataset, download the dataset and extract the csv files to `data/mimic3/raw/` and `data/mimic4/raw/` in this project.

After the datasets are downloaded, please put each of them into a specified directory of the project.

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



### (2) Preprocess datasets 

````python
For the MIMIC-III dataset: 

```bash
python run_preprocess_mimic-iii.py
```

For the MIMIC-IV dataset: 

```
python run_preprocess_mimic-iv.py
```
````

## 2. Configuration

Please see `train_hyper.py` for detailed configurations.



## 3. Train model

You can train the model with the following command:

````python
```bash
python train_hyper.py
```
````



## Acknowledgement

If this work is useful in your research, please cite our paper.

```python
@inproceedings{zhang2024fusion, 
  title={Fusion of Dynamic Hypergraph and Clinical Event for Sequential Diagnosis Prediction},
  author={Zhang, Xin and Peng, Xueping and Guan, Hongjiao and Zhao, Long and Qiao, Xinxiao and Lu, Wenpeng },
  booktitle={Proceedings of the 29th IEEE International Conference on Parallel and Distributed Systems (ICPADS)},
  year={2024}}
```
