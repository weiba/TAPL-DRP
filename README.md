TAPL-DRP
===============================
Source code and data for "TAPL-DRP: A Transfer Learning Approach for Predicting Patient Anticancer Drug Response"

# Requirements
All implementations of TAPL-DRP are based on PyTorch. TAPL-DRP requires the following dependencies:
- python==3.10.16
- torch==2.2.2
- torch-geometric==2.6.1
- numpy==1.26.4
- scipy==1.13.1
- pandas==2.2.3
- scikit-learn==1.6.1
- matplotlib==3.10.1
- seaborn==0.13.2
- imbalanced-learn==0.13.0
- rdkit==2024.9.6
- networkx==3.3
- tqdm==4.67.1

## Installation
You can install the required dependencies using pip:
```bash
pip install -r requirements.txt
```

Or install them individually:
```bash
pip install torch==2.2.2 torch-geometric==2.6.1 numpy==1.26.4 scipy==1.13.1 pandas==2.2.3 scikit-learn==1.6.1 matplotlib==3.10.1 seaborn==0.13.2 imbalanced-learn==0.13.0 rdkit==2024.9.6 networkx==3.3 tqdm==4.67.1
```

# Data
- **data/TCGA** records training data, test data, and labeling related to the five drugs associated with TCGA.
- **data/PDTC** records training data, test data, and labeling related to the drugs associated with PDTC.
- **data/ccle_sample_info.csv** records biological information related to CCLE samples.
- **data/pretrain_ccle.csv** records gene expression data from unlabeled CCLE samples.
- **data/pretrain_tcga.csv** records gene expression data from unlabeled TCGA samples.
- **data/pdtc_uq1000_feature.csv** records gene expression data from unlabeled PDTC samples.
- **data/GDSC1_fitted_dose_response_25Feb20.csv** and **data/GDSC2_fitted_dose_response_25Feb20.csv** records data on drug use and response in GDSC samples.
- **data/DrugResponsesAUCModels.txt** records response data for PDTC sample-drug pairs.
- **data/pdtc_gdsc_drug_mapping.csv** records the drug names associated with PDTC and their SMILES.
- **data/uq1000_feature.csv** records gene expression data for unlabeled TCGA samples and CCLE samples.
- **data/xena_sample_info_df.csv** records biological information related to TCGA samples.

# Code Structure
- **tools/model.py** defines the model architectures used in the training process.
- **tools/dataprocess.py** defines data preprocessing utilities.
- **drugmodels/** contains drug representation models (GIN, GCN, GAT).
- **data.py** defines the data loading and preprocessing for the model.
- **precontext.py** defines drug pretraining functionality.
- **pretrain.py** defines the training of the domain invariant feature extraction phase.
- **classifier.py** defines the classifier training of the model.
- **train_all.py** provides a unified training pipeline.

## Preprocessing your own data
To process your own data and prepare it for TAPL-DRP:

> In our study, the source cell line data and target patient data follow the format from [codeae](https://codeocean.com/capsule/1993810/tree/v1)[1]. You can run our program with your own data by processing the data into source and target domain data of the same dimensions, with one-dimensional labeled data for each sample-drug pair. You can refer to the data format in **data/TCGA**.
> 
> [1] He, Di, et al. "A context-aware deconfounding autoencoder for robust prediction of personalized clinical drug response from cell-line compound screening." Nature Machine Intelligence 4.10 (2022): 879-892.

# Usage

## Quick Start
Once you have configured the environment, you can run **TAPL-DRP** using our provided data:

```bash
python train_all.py
```

This will run the complete training pipeline with default settings (10 runs on both PDTC and TCGA datasets).

**Note:** The current implementation runs exactly 10 iterations by default. If you need to modify the number of runs, you can edit the `range(0, 10)` in `train_all.py`.

### Step-by-step training:
```bash
# Step 1: Pretrain the domain adaptation model (runs 10 iterations)
python pretrain.py

# Step 2: Train the classifier (runs 10 iterations)
python classifier.py
```

## Training Process
To build and evaluate our model, we use cell line gene expression data as the source domain and patient gene expression data as the target domain. We divide the source domain data into five folds for cross-validation, with a training set and validation set ratio of 4:1, and the test set uses the target domain data. We use grid search to determine the best parameters and retain the model with the best performance on the validation set in each fold. We also evaluate the classifier performance on each test set. Finally, we use the average AUC and AUPRC of the five folds on the test set as our single-run metrics, and then take the average of 10 runs as the final metric.

> In **pretrain.py** we use a cosine annealing strategy with a learning rate set to 0.001, and train epochs by grid searching to store models at different pre-training epochs. We use an early stopping strategy for the model when the validation set loss does not decrease for 50 consecutive times without generative adversarial training and when the validation set loss does not decrease for 20 consecutive times with generative adversarial training.

> In **classifier.py** we train the classifier at this stage by performing a parameter search, testing for the use of a cosine annealing strategy while lr=[0.01, 0.001]. We use the early stopping strategy when the AUC value on the validation set does not increase for 20 consecutive times. We predict model performance at this stage.

## Using your own data
Alternatively, you can run our program with your own data and custom settings:

```bash
# Step 1: Pretrain with your data
python pretrain.py \
--outfolder path/to/folder_to_save_pretrain_models \
--source path/to/your_pretrain_source_data.csv \
--target path/to/your_pretrain_target_data.csv

# Step 2: Train classifier with your data
python classifier.py \
--dataset other \
--data path/to/your_data_folder \
--drug path/to/your_drug_name.csv \
--drugpth path/to/drug_encoder.pth \
--pretrain_model path/to/your_pretrain_models_path \
--outfolder path/to/save_result_and_others \
--outname result_file_name.csv
```

**Note:** 
> You need to ensure that the data dimensions of your source and target domains are the same.

> The **your_data_folder** is a folder that contains many medication folders while each medication folder contains sourcedata.csv, targetdata.csv, sourcelabel.csv, targetlabel.csv. The format of each file can be referred to **data/TCGA**.

# Results
The training results will be saved in the **result/** directory:
- **result/pretrain/** contains pretrained models
- **result/prototypical/** contains classification results
- **time_txt/** contains detailed metrics for each run

# Contact
If you have any questions regarding our code or data, please do not hesitate to open an issue or directly contact us.
