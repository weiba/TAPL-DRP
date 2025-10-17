import logging
import pandas as pd
import numpy as np
import os
import torch
from imblearn.over_sampling import SMOTE
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.utils import resample
from torch.utils.data import TensorDataset, DataLoader
from sklearn.preprocessing import LabelEncoder
from scipy.stats import pearsonr
from torch.utils.data import WeightedRandomSampler

device = 'cuda' if torch.cuda.is_available() else 'cpu'
print('use device:', device)

if not os.path.exists('logs'):
    os.makedirs('logs')

logging.basicConfig(
    filename='logs/pretrain_data.log',
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

print("Loading data...")

def pretrain_data():
    ccle_df = pd.read_csv(os.path.join('data', 'pretrain_ccle.csv'), index_col=0, header=0)
    xena_df = pd.read_csv(os.path.join('data', 'pretrain_tcga.csv'), index_col=0, header=0)
    ccle_tissue_df = pd.read_csv(os.path.join('data', 'CCLE_Tissue_no_engineered_add_ACH_001316.csv'), index_col='CELL_Line')
    tcga_tissue_df = pd.read_csv(os.path.join('data', 'TCGA_Tissue.csv'), index_col='TCGA')
    ccle_sample_info_df = pd.read_csv(os.path.join('data', 'ccle_sample_info_df.csv'), index_col=0, header=0)
    xena_sample_info_df = pd.read_csv(os.path.join('data', 'xena_sample_info_df.csv'), index_col=0, header=0)

    ccle_df = ccle_df.loc[ccle_df.index.intersection(ccle_tissue_df.index).intersection(ccle_sample_info_df.index)]
    xena_df = xena_df.loc[xena_df.index.intersection(tcga_tissue_df.index).intersection(xena_sample_info_df.index)]

    excluded_ccle_samples = []
    excluded_ccle_samples.extend(ccle_df.index.difference(ccle_sample_info_df.index))
    excluded_ccle_diseases = ccle_sample_info_df['primary_disease'].value_counts()[
        ccle_sample_info_df['primary_disease'].value_counts() < 2].index
    excluded_ccle_samples.extend(
        ccle_sample_info_df[ccle_sample_info_df['primary_disease'].isin(excluded_ccle_diseases)].index)
    to_split_ccle_df = ccle_df[~ccle_df.index.isin(excluded_ccle_samples)]

    logging.info("CCLE disease distribution before split:")
    logging.info(ccle_sample_info_df.loc[to_split_ccle_df.index, 'primary_disease'].value_counts())

    train_ccle_df, test_ccle_df = train_test_split(
        to_split_ccle_df,
        test_size=0.1,
        stratify=ccle_sample_info_df.loc[to_split_ccle_df.index, 'primary_disease']
    )
    test_ccle_df = pd.concat([test_ccle_df, ccle_df.loc[excluded_ccle_samples]])

    ccle_labels = ccle_tissue_df.loc[train_ccle_df.index, 'Tissue_name']
    ccle_label_counts = ccle_labels.value_counts()
    tcga_labels = tcga_tissue_df.loc[xena_df.index, 'Tissue_name']
    tcga_label_counts = tcga_labels.value_counts()

    common_labels = ccle_label_counts.index.intersection(tcga_label_counts.index)
    logging.info(f"Common tissue types between CCLE and TCGA: {common_labels.tolist()}")
    logging.info(f"CCLE exclusive tissue types: {ccle_label_counts.index.difference(tcga_label_counts.index).tolist()}")

    train_xena_df = pd.DataFrame(columns=xena_df.columns)
    for label in common_labels:
        label_mask = tcga_labels == label
        label_samples = xena_df[label_mask]
        target_count = ccle_label_counts[label]
        
        if len(label_samples) < target_count:
            logging.info(f"Upsampling TCGA tissue type {label}: {len(label_samples)} -> {target_count}")
            label_samples = resample(label_samples, replace=True, n_samples=target_count, random_state=2020)
        elif len(label_samples) > target_count:
            logging.info(f"Downsampling TCGA tissue type {label}: {len(label_samples)} -> {target_count}")
            label_samples = resample(label_samples, replace=False, n_samples=target_count, random_state=2020)
        
        train_xena_df = pd.concat([train_xena_df, label_samples])

    remaining_xena_df = xena_df.loc[~xena_df.index.isin(train_xena_df.index)]
    if len(remaining_xena_df) > 0:
        test_xena_df = train_test_split(
            remaining_xena_df,
            test_size=len(test_ccle_df) / len(xena_df),
            stratify=xena_sample_info_df.loc[remaining_xena_df.index, '_primary_disease'],
            random_state=2020
        )[1]
    else:
        logging.warning("Not enough TCGA samples for test set, sampling from xena_df")
        test_xena_df = resample(xena_df, n_samples=len(test_ccle_df), random_state=2020)

    missing_indices = train_xena_df.index[~train_xena_df.index.isin(xena_sample_info_df.index)]
    if len(missing_indices) > 0:
        logging.warning(f"Found {len(missing_indices)} indices in train_xena_df not present in xena_sample_info_df: {missing_indices.tolist()}")

    logging.info("CCLE disease distribution in train_ccle_df:")
    logging.info(ccle_sample_info_df.loc[train_ccle_df.index, 'primary_disease'].value_counts())
    logging.info("TCGA disease distribution in train_xena_df after resampling:")
    logging.info(xena_sample_info_df.loc[train_xena_df.index.intersection(xena_sample_info_df.index), '_primary_disease'].value_counts())

    le = LabelEncoder()
    all_labels = pd.concat([ccle_tissue_df['Tissue_name'], tcga_tissue_df['Tissue_name']])
    le.fit(all_labels)
    unique_labels = list(range(len(le.classes_)))
    
    ccle_labels = torch.tensor(le.transform(ccle_labels)).to(device)
    tcga_labels = torch.tensor(le.transform(tcga_tissue_df.loc[train_xena_df.index.intersection(tcga_tissue_df.index), 'Tissue_name'])).to(device)
    ccle_test_labels = torch.tensor(le.transform(ccle_tissue_df.loc[test_ccle_df.index.intersection(ccle_tissue_df.index), 'Tissue_name'])).to(device)
    tcga_test_labels = torch.tensor(le.transform(tcga_tissue_df.loc[test_xena_df.index.intersection(tcga_tissue_df.index), 'Tissue_name'])).to(device)

    ccle_tensor = torch.from_numpy(train_ccle_df.values).type(torch.float32).to(device)
    ccle_test_tensor = torch.from_numpy(test_ccle_df.values).type(torch.float32).to(device)
    tcga_tensor = torch.from_numpy(train_xena_df.values).type(torch.float32).to(device)
    tcga_test_tensor = torch.from_numpy(test_xena_df.values).type(torch.float32).to(device)

    batch_size = min(64, len(train_ccle_df) // 4)
    ccle_weights = torch.ones(len(ccle_labels)).to(device)
    tcga_weights = torch.ones(len(tcga_labels)).to(device)
    ccle_sampler = WeightedRandomSampler(ccle_weights, len(ccle_weights), replacement=True)
    tcga_sampler = WeightedRandomSampler(tcga_weights, len(tcga_weights), replacement=True)

    ccle_dataset = TensorDataset(ccle_tensor, ccle_labels)
    ccle_loader = DataLoader(ccle_dataset, batch_size=batch_size, sampler=ccle_sampler, drop_last=True)
    tcga_dataset = TensorDataset(tcga_tensor, tcga_labels)
    tcga_loader = DataLoader(tcga_dataset, batch_size=batch_size, sampler=tcga_sampler, drop_last=True)
    ccle_test_dataset = TensorDataset(ccle_test_tensor, ccle_test_labels)
    tcga_test_dataset = TensorDataset(tcga_test_tensor, tcga_test_labels)

    logging.info(f"CCLE train: {len(train_ccle_df)}, test: {len(test_ccle_df)}")
    logging.info(f"TCGA train: {len(train_xena_df)}, test: {len(test_xena_df)}")
    return (ccle_loader, ccle_test_dataset), (tcga_loader, tcga_test_dataset), unique_labels, batch_size

def pretrain_loader(df:pd.DataFrame):
    train_df, test_df = train_test_split(df, test_size=0.2)
    train_tensor = torch.from_numpy(train_df.values).type(torch.float32).to(device)
    test_tensor = torch.from_numpy(test_df.values).type(torch.float32).to(device)
    batch_size = 64
    train_dataset = TensorDataset(train_tensor)
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, drop_last=True)
    test_dataset = TensorDataset(test_tensor)
    test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True, drop_last=True)
    return train_dataloader, test_dataloader

def PDTC_source_5fold(drug):
    measurement = 'Z_SCORE'
    threshold = 0.0
    drugs_to_keep = [drug.lower()]
    gdsc_target_file1 = os.path.join('data', 'GDSC1_fitted_dose_response_25Feb20.csv')
    gdsc_target_file2 = os.path.join('data', 'GDSC2_fitted_dose_response_25Feb20.csv')
    gdsc1_response = pd.read_csv(gdsc_target_file1)
    gdsc2_response = pd.read_csv(gdsc_target_file2)
    gdsc1_sensitivity_df = gdsc1_response[['COSMIC_ID', 'DRUG_NAME', measurement]]
    gdsc2_sensitivity_df = gdsc2_response[['COSMIC_ID', 'DRUG_NAME', measurement]]
    gdsc1_sensitivity_df.loc[:, 'DRUG_NAME'] = gdsc1_sensitivity_df['DRUG_NAME'].str.lower()
    gdsc2_sensitivity_df.loc[:, 'DRUG_NAME'] = gdsc2_sensitivity_df['DRUG_NAME'].str.lower()

    if measurement == 'LN_IC50':
        gdsc1_sensitivity_df.loc[:, measurement] = np.exp(gdsc1_sensitivity_df[measurement])
        gdsc2_sensitivity_df.loc[:, measurement] = np.exp(gdsc2_sensitivity_df[measurement])

    gdsc1_sensitivity_df = gdsc1_sensitivity_df.loc[gdsc1_sensitivity_df.DRUG_NAME.isin(drugs_to_keep)]
    gdsc2_sensitivity_df = gdsc2_sensitivity_df.loc[gdsc2_sensitivity_df.DRUG_NAME.isin(drugs_to_keep)]
    gdsc1_target_df = gdsc1_sensitivity_df.groupby(['COSMIC_ID', 'DRUG_NAME']).mean()
    gdsc2_target_df = gdsc2_sensitivity_df.groupby(['COSMIC_ID', 'DRUG_NAME']).mean()
    gdsc1_target_df = gdsc1_target_df.loc[gdsc1_target_df.index.difference(gdsc2_target_df.index)]
    gdsc_target_df = pd.concat([gdsc1_target_df, gdsc2_target_df])
    target_df = gdsc_target_df.reset_index().pivot_table(values=measurement, index='COSMIC_ID', columns='DRUG_NAME')
    ccle_sample_file = os.path.join('data', 'ccle_sample_info.csv')
    ccle_sample_info = pd.read_csv(ccle_sample_file, index_col=4)
    ccle_sample_info = ccle_sample_info.loc[ccle_sample_info.index.dropna()]
    ccle_sample_info.index = ccle_sample_info.index.astype('int')
    gdsc_sample_file = os.path.join('data', 'gdsc_cell_line_annotation.csv')
    gdsc_sample_info = pd.read_csv(gdsc_sample_file, header=0, index_col=1)
    gdsc_sample_info = gdsc_sample_info.loc[gdsc_sample_info.index.dropna()]
    gdsc_sample_info.index = gdsc_sample_info.index.astype('int')
    # gdsc_sample_info = gdsc_sample_info.loc[gdsc_sample_info.iloc[:, 8].dropna().index]
    gdsc_sample_mapping = gdsc_sample_info.merge(ccle_sample_info, left_index=True, right_index=True, how='inner')[
        ['DepMap_ID']]
    gdsc_sample_mapping_dict = gdsc_sample_mapping.to_dict()['DepMap_ID']
    target_df.index = target_df.index.map(gdsc_sample_mapping_dict)
    target_df = target_df.loc[target_df.index.dropna()]
    ccle_target_df = target_df[drugs_to_keep[0]]
    ccle_target_df.dropna(inplace=True)
    gex_feature_file = os.path.join('data', 'uq1000_feature.csv')
    gex_features_df = pd.read_csv(gex_feature_file, index_col=0)
    ccle_labeled_samples = gex_features_df.index.intersection(ccle_target_df.index)

    if threshold is None:
        threshold = np.median(ccle_target_df.loc[ccle_labeled_samples])

    ccle_labels = (ccle_target_df.loc[ccle_labeled_samples] < threshold).astype('int')
    ccle_labeled_feature_df = gex_features_df.loc[ccle_labeled_samples]
    assert all(ccle_labels.index == ccle_labeled_feature_df.index)
    s_kfold = StratifiedKFold(n_splits=5, random_state=2020, shuffle=True)
    for train_index, test_index in s_kfold.split(ccle_labeled_feature_df.values, ccle_labels.values):
        train_labeled_ccle_df, test_labeled_ccle_df = ccle_labeled_feature_df.values[train_index], \
                                                      ccle_labeled_feature_df.values[test_index]
        train_ccle_labels, test_ccle_labels = ccle_labels.values[train_index], ccle_labels.values[test_index]
        # df->tensor
        ccle_train_data = torch.from_numpy(train_labeled_ccle_df).type(torch.float32).to(device)
        ccle_train_label = torch.from_numpy(train_ccle_labels).type(torch.float32).squeeze().to(device)
        ccle_test_data = torch.from_numpy(test_labeled_ccle_df).type(torch.float32).to(device)
        ccle_test_label = torch.from_numpy(test_ccle_labels).type(torch.float32).squeeze().to(device)

        yield (ccle_train_data, ccle_train_label), (ccle_test_data, ccle_test_label)

def PDTC_target_data(drug):
    pdtc_gex_file = os.path.join('data', 'pdtc_uq1000_feature.csv')
    pdtc_features_df = pd.read_csv(pdtc_gex_file, index_col=0)
    pdtc_target_file = os.path.join('data', 'DrugResponsesAUCModels.txt')
    target_df = pd.read_csv(pdtc_target_file, index_col=0, sep='\t')
    drug_target_df = target_df.loc[target_df.Drug == drug]
    labeled_samples = drug_target_df.index.intersection(pdtc_features_df.index)
    drug_target_vec = drug_target_df.loc[labeled_samples, 'AUC']
    drug_feature_df = pdtc_features_df.loc[labeled_samples]
    threshold = np.median(drug_target_vec)
    drug_label_vec = (drug_target_vec < threshold).astype('int')
    pdtc_features = torch.from_numpy(drug_feature_df.values).type(torch.float32).to(device)
    pdtc_label = torch.from_numpy(drug_label_vec.values).type(torch.float32).squeeze().to(device)
    return (pdtc_features, pdtc_label)

def PDTC_data_generator(drug):
    drug_mapping_df = pd.read_csv(os.path.join('data', 'pdtc_gdsc_drug_mapping.csv'), index_col=0)
    drug_name = drug_mapping_df.loc[drug, 'drug_name']
    pdtc_data = PDTC_target_data(drug_name)
    gdsc_drug = drug_mapping_df.loc[drug, 'gdsc_name']
    ccle_data_tuple = PDTC_source_5fold(gdsc_drug)
    for ccle_train_data, ccle_eval_data in ccle_data_tuple:
        yield (ccle_train_data, ccle_eval_data, pdtc_data)


def TCGA_source_5fold(drug):
    # data df gene_num:1426
    ccle_features_df = pd.read_csv(os.path.join('data','TCGA', drug + 'data', 'ccledata.csv'), index_col=0, header=0)
    ccle_label_df = pd.read_csv(os.path.join('data','TCGA', drug + 'data', 'cclelabel.csv'), index_col=0, header=0)
    # split 5-fold
    s_kfold = StratifiedKFold(n_splits=5, random_state=2020, shuffle=True)
    for train_index, test_index in s_kfold.split(ccle_features_df.values, ccle_label_df.values):
        train_labeled_ccle_df, test_labeled_ccle_df = ccle_features_df.values[train_index], \
                                                    ccle_features_df.values[test_index]
        train_ccle_labels, test_ccle_labels = ccle_label_df.values[train_index], ccle_label_df.values[test_index]
        # df->tensor
        ccle_train_data = torch.from_numpy(train_labeled_ccle_df).type(torch.float32).to(device)
        ccle_train_label = torch.from_numpy(train_ccle_labels).type(torch.float32).squeeze().to(device)
        ccle_test_data = torch.from_numpy(test_labeled_ccle_df).type(torch.float32).to(device)
        ccle_test_label = torch.from_numpy(test_ccle_labels).type(torch.float32).squeeze().to(device)

        yield (ccle_train_data, ccle_train_label), (ccle_test_data, ccle_test_label)


def TCGA_target_data(drug):
    tcga_features_df = pd.read_csv(os.path.join('data','TCGA', drug + 'data', 'tcgadata.csv'), index_col=0, header=0)
    # tcga_features_df = tcga_features_df.reindex(columns=ccle_columns)
    tcga_label_df = pd.read_csv(os.path.join('data','TCGA', drug + 'data', 'tcgalabel.csv'), index_col=0, header=0)
    tcga_features = torch.from_numpy(tcga_features_df.values).type(torch.float32).to(device)
    tcga_label = torch.from_numpy(tcga_label_df.values).type(torch.float32).squeeze().to(device)

    return (tcga_features, tcga_label)


def TCGA_data_generator(drug):
    tcga_data = TCGA_target_data(drug)
    ccle_data_tuple = TCGA_source_5fold(drug)
    for ccle_train_data, ccle_eval_data in ccle_data_tuple:
        yield (ccle_train_data, ccle_eval_data, tcga_data)


def other_data_generator(datafolder):
    os.path.join(datafolder)
    source_features_df = pd.read_csv(os.path.join(datafolder, 'sourcedata.csv'), index_col=0, header=0)
    source_label_df = pd.read_csv(os.path.join(datafolder, 'sourcelabel.csv'), index_col=0, header=0)
    target_features_df = pd.read_csv(os.path.join(datafolder, 'targetdata.csv'), index_col=0, header=0)
    target_label_df = pd.read_csv(os.path.join(datafolder, 'targetlabel.csv'), index_col=0, header=0)
    s_kfold = StratifiedKFold(n_splits=5, shuffle=True)
    for train_index, test_index in s_kfold.split(source_features_df.values, source_label_df.values):
        train_labeled_source_df, test_labeled_source_df = source_features_df.values[train_index], \
                                                    source_features_df.values[test_index]
        train_source_labels, test_source_labels = source_label_df.values[train_index], source_label_df.values[test_index]
        # df->tensor
        source_train_data = torch.from_numpy(train_labeled_source_df).type(torch.float32).to(device)
        source_train_label = torch.from_numpy(train_source_labels).type(torch.float32).squeeze().to(device)
        source_test_data = torch.from_numpy(test_labeled_source_df).type(torch.float32).to(device)
        source_test_label = torch.from_numpy(test_source_labels).type(torch.float32).squeeze().to(device)
        target_data = torch.from_numpy(target_features_df.values).type(torch.float32).to(device)
        target_label = torch.from_numpy(target_label_df.values).type(torch.float32).squeeze().to(device)

        yield (source_train_data, source_train_label),(source_test_data, source_test_label),(target_data, target_label)