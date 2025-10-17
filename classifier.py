import argparse
import os
import torch
import math
import copy
import json
import itertools
import pandas as pd
import numpy as np
import torch.nn as nn
import torch.autograd as autograd
from data import *
from copy import deepcopy
from itertools import chain
from collections import defaultdict
from precontext import drug_pretrain
from tools.dataprocess import *
from torch_geometric import data as DATA
from tools.model import *
from drugmodels.ginconv import GINConvNet
from sklearn.metrics import accuracy_score, f1_score, precision_recall_curve, average_precision_score, roc_auc_score, matthews_corrcoef, precision_score, recall_score

device = 'cuda' if torch.cuda.is_available() else 'cpu'
print('use device:', device)

def safemakedirs(folder):
    if os.path.exists(folder) == False:
        os.makedirs(folder)

def fine_tune(Data, encoder, classifymodel, drugmodel, optimizer, scheduler, drug_data, drug_folder, kfold_num, param1, param2):
    classification_loss = nn.BCEWithLogitsLoss()
    num_epoch = param2['train_num_epochs']
    best_metrics = {
        'EPOCH': 0,
        'AUC': 0,
        'AUPRC': 0,
        'F1': 0,
        'Accuracy': 0,
        'MCC': 0,
        'Precision': 0,
        'Recall': 0
    }
    best_test_metrics = {
        'EPOCH': 0,
        'AUC': 0,
        'AUPRC': 0,
        'F1': 0,
        'Accuracy': 0,
        'MCC': 0,
        'Precision': 0,
        'Recall': 0
    }
    source_train_data, source_test_data, target_data = Data[0], Data[1], Data[2]
    encoder.eval()
    classifymodel.train()
    drugmodel.eval()
    loss_log_name = os.path.join(drug_folder, str(kfold_num) + 'train_loss_log.txt')
    eval_log_name = os.path.join(drug_folder, str(kfold_num) + '_foldeval.txt')
    test_log_name = os.path.join(drug_folder, str(kfold_num) + '_foldtest.txt')
    best_eval_log_name = os.path.join(drug_folder, str(kfold_num) + "_fold_best_auc.txt")
    train_feature_name = os.path.join(drug_folder, str(kfold_num) + '_fold_trainfeature')
    eval_feature_name = os.path.join(drug_folder, str(kfold_num) + '_fold_evalfeature')
    test_feature_name = os.path.join(drug_folder, str(kfold_num) + '_fold_testfeature')
    tolerance = 0
    max_tolerance = 20
    for epoch in range(num_epoch):
        finetune_lossdict = defaultdict(float)
        optimizer.zero_grad()
        _, z, _, _ = encoder(source_train_data[0])
        drugemb = drugmodel(drug_data)
        scatemb = cat_tensor_with_drug(z, drugemb)
        predict = classifymodel(scatemb)
        c_loss = classification_loss(predict, source_train_data[1])
        loss = c_loss
        loss.backward()
        optimizer.step()
        if param1['scheduler_flag'] == True:
            scheduler.step()
        if torch.is_tensor(c_loss):
            c_loss = c_loss.item()
        finetune_lossdict.update({'class_loss': c_loss})
        append_file(loss_log_name, finetune_lossdict)
        with torch.no_grad():
            _, evalemb, _, _ = encoder(source_test_data[0])
            _, testemb, _, _ = encoder(target_data[0])
            catemb_eval = cat_tensor_with_drug(evalemb, drugemb)
            catemb_test = cat_tensor_with_drug(testemb, drugemb)
            _, trainemb, _, _ = encoder(source_train_data[0])
            catemb_train = cat_tensor_with_drug(trainemb, drugemb)
            test_y_pred = classifymodel(catemb_test).cpu().detach().numpy()
            eval_y_pred = classifymodel(catemb_eval).cpu().detach().numpy()
            eval_y_true = source_test_data[1].cpu().detach().numpy()
            test_y_true = target_data[1].cpu().detach().numpy()
            eval_auc = roc_auc_score(eval_y_true, eval_y_pred)
            eval_auprc = average_precision_score(eval_y_true, eval_y_pred)
            eval_y_pred_binary = (eval_y_pred > 0.5).astype('int')
            eval_f1 = f1_score(eval_y_true, eval_y_pred_binary)
            eval_acc = accuracy_score(eval_y_true, eval_y_pred_binary)
            eval_mcc = matthews_corrcoef(eval_y_true, eval_y_pred_binary)
            eval_precision = precision_score(eval_y_true, eval_y_pred_binary, zero_division=0)
            eval_recall = recall_score(eval_y_true, eval_y_pred_binary, zero_division=0)
            eval_metrics = {
                'EPOCH': epoch,
                'AUC': eval_auc,
                'AUPRC': eval_auprc,
                'F1': eval_f1,
                'Accuracy': eval_acc,
                'MCC': eval_mcc,
                'Precision': eval_precision,
                'Recall': eval_recall
            }
            append_file(eval_log_name, eval_metrics)
            test_auc = roc_auc_score(test_y_true, test_y_pred)
            test_auprc = average_precision_score(test_y_true, test_y_pred)
            test_y_pred_binary = (test_y_pred > 0.5).astype('int')
            test_f1 = f1_score(test_y_true, test_y_pred_binary)
            test_acc = accuracy_score(test_y_true, test_y_pred_binary)
            test_mcc = matthews_corrcoef(test_y_true, test_y_pred_binary)
            test_precision = precision_score(test_y_true, test_y_pred_binary, zero_division=0)
            test_recall = recall_score(test_y_true, test_y_pred_binary, zero_division=0)
            test_metrics = {
                'EPOCH': epoch,
                'AUC': test_auc,
                'AUPRC': test_auprc,
                'F1': test_f1,
                'Accuracy': test_acc,
                'MCC': test_mcc,
                'Precision': test_precision,
                'Recall': test_recall
            }
            append_file(test_log_name, test_metrics)
            if eval_metrics['AUC'] >= best_metrics['AUC']:
                best_metrics.update(eval_metrics)
                best_metrics['EPOCH'] = epoch
                best_test_metrics.update(test_metrics)
                best_test_metrics['EPOCH'] = epoch
                temp_log = {'epoch': epoch, "eval auc=": eval_metrics['AUC'], "test auc=": test_metrics['AUC']}
                append_file(best_eval_log_name, temp_log)
                tolerance = 0
                best_train_feature = catemb_train
                best_eval_feature = catemb_eval
                best_test_feature = catemb_test
                best_classifier = copy.deepcopy(classifymodel)
            else:
                tolerance += 1
                if tolerance in (10, 20, 50):
                    append_file(best_eval_log_name, {'early stop': tolerance})
            if tolerance >= max_tolerance:
                break
    torch.save(best_train_feature, train_feature_name)
    torch.save(best_eval_feature, eval_feature_name)
    torch.save(best_test_feature, test_feature_name)
    torch.save(best_classifier.state_dict(), os.path.join(drug_folder, str(kfold_num) + 'fold_classifier.pth'))
    print('{}_fold feature and classifier saved , best_test_auc:{}'.format(kfold_num, best_test_metrics['AUC']))
    return best_test_metrics

def step_1_finetune(parent_folder, drug_list, drug_smiles, datatype, outfolder, resultname, drugpth, otherfolder=None):
    drug_encoder_dict = drugpth
    params_grid = {
        "pretrain_num_epochs": [0, 100, 300],
        'pretrain_learning_rate': [0.001],
        'gan_learning_rate': [0.001],
        "train_num_epochs": [100, 200, 300, 500, 750, 1000, 1500, 2000, 2500, 3000]
    }
    keys, values = zip(*params_grid.items())
    update_params_dict_list = [dict(zip(keys, v)) for v in itertools.product(*values)]
    fine_tune_params_grid = {
        'ftlr' : [0.01, 0.001],
        'scheduler_flag' : [True, False]
    }
    ftkeys, ftvalues = zip(*fine_tune_params_grid.items())
    fine_tune_dict_list = [dict(zip(ftkeys, v)) for v in itertools.product(*ftvalues)]
    all_metrics = {}
    for drug in drug_list:
        all_metrics.update({
            drug+'auc': 0,
            drug+'AUPRC': 0,
            drug+'F1': 0,
            drug+'Accuracy': 0,
            drug+'MCC': 0,
            drug+'Precision': 0,
            drug+'Recall': 0,
            drug+'folder': None
        })
    best_df = pd.DataFrame(index=drug_list, columns=['auc', 'aucvar', 'auprc', 'auprcvar', 'f1', 'f1var', 'acc', 'accvar', 'mcc', 'mccvar', 'precision', 'precisionvar', 'recall', 'recallvar'])
    for fine_tune_dict in fine_tune_dict_list:
        for param in update_params_dict_list:
            for drug, drug_smile in zip(drug_list, drug_smiles):
                set_dir_name = 'pt_epochs_' + str(param['pretrain_num_epochs']) + \
                               ',t_epochs_' + str(param['train_num_epochs']) + \
                               ',Ptlr_' + str(param['pretrain_learning_rate']) + \
                               ',tlr' + str(param['gan_learning_rate'])
                model_folder = os.path.join(parent_folder, set_dir_name)
                encoder_state_dict = torch.load(os.path.join(model_folder, 'after_traingan_shared_vae.pth'))
                print('train drug:', drug)
                _, x, edge_index = smile_to_graph(drug_smile)
                x = torch.tensor(np.array(x), device=device).float()
                edge_index = torch.tensor(edge_index, device=device).t()
                drug_data = DATA.Data(
                    x=x,
                    edge_index=edge_index
                )
                if datatype == 'PDTC':
                    data_generator = PDTC_data_generator(drug)
                elif datatype == 'TCGA':
                    data_generator = TCGA_data_generator(drug)
                else:
                   Ora_generator = other_data_generator(os.path.join(otherfolder, drug))
                auc_folder = os.path.join(model_folder, 'feature_save')
                drug_auc_folder = os.path.join(auc_folder, drug)
                safemakedirs(drug_auc_folder)
                test_auc_list = []
                addauc = []
                addauprc = []
                addf1 = []
                addacc = []
                addmcc = []
                addprecision = []
                addrecall = []
                i = 0

                for data in data_generator:
                    temp_folder = os.path.join(drug_auc_folder,"ftepoch"+str(param['train_num_epochs'])+",lr:"+str(fine_tune_dict['ftlr'])+",CosAL:"+str(fine_tune_dict['scheduler_flag']))
                    log_folder = os.path.join(temp_folder, 'log')
                    safemakedirs(log_folder)
                    test_auc_log_name = os.path.join(temp_folder, 'step1_test_auc.txt')
                    temp_encoder = VAE(input_size=1426, output_size=1426, latent_size=32, hidden_size=128).to(device)
                    temp_encoder.load_state_dict(encoder_state_dict)
                    classifymodel = Classify(input_dim=32+10).to(device)
                    #print("cxcxcxc:",drug_data.x.shape)
                    drug_gcnmodel = GINConvNet(input_dim=drug_data.x.shape[1], output_dim=10).to(device)
                    drug_gcnmodel.load_state_dict(drug_encoder_dict)
                    fine_tune_optimizer = torch.optim.AdamW(classifymodel.parameters(), lr=fine_tune_dict['ftlr'])
                    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(fine_tune_optimizer, param['train_num_epochs'])
                    test_history = fine_tune(
                        Data = data,
                        encoder=temp_encoder,
                        drugmodel=drug_gcnmodel,
                        classifymodel=classifymodel,
                        optimizer = fine_tune_optimizer,
                        scheduler = scheduler,
                        drug_data=drug_data,
                        drug_folder=log_folder,
                        kfold_num = i,
                        param1 = fine_tune_dict,
                        param2 = param
                    )
                    test_auc_list.append(test_history)
                    addauc.append(test_history['AUC'])
                    addauprc.append(test_history['AUPRC'])
                    addf1.append(test_history['F1'])
                    addacc.append(test_history['Accuracy'])
                    addmcc.append(test_history['MCC'])
                    addprecision.append(test_history['Precision'])
                    addrecall.append(test_history['Recall'])
                    i += 1
                    
                    if i == 5:
                        meanauc = sum(addauc) / len(addauc)
                        meanauprc = sum(addauprc) / len(addauprc)
                        meanf1 = sum(addf1) / len(addf1)
                        meanacc = sum(addacc) / len(addacc)
                        meanmcc = sum(addmcc) / len(addmcc)
                        meanprecision = sum(addprecision) / len(addprecision)
                        meanrecall = sum(addrecall) / len(addrecall)
                        if meanauc > all_metrics[drug+'auc']:
                            all_metrics[drug+'auc'] = meanauc
                            all_metrics[drug+'AUPRC'] = meanauprc
                            all_metrics[drug+'F1'] = meanf1
                            all_metrics[drug+'Accuracy'] = meanacc
                            all_metrics[drug+'MCC'] = meanmcc
                            all_metrics[drug+'Precision'] = meanprecision
                            all_metrics[drug+'Recall'] = meanrecall
                            all_metrics[drug+'folder'] = temp_folder
                            best_df.at[drug, 'auc'] = meanauc
                            best_df.at[drug, 'aucvar'] = np.var(addauc)
                            best_df.at[drug, 'auprc'] = meanauprc
                            best_df.at[drug, 'auprcvar'] = np.var(addauprc)
                            best_df.at[drug, 'f1'] = meanf1
                            best_df.at[drug, 'f1var'] = np.var(addf1)
                            best_df.at[drug, 'acc'] = meanacc
                            best_df.at[drug, 'accvar'] = np.var(addacc)
                            best_df.at[drug, 'mcc'] = meanmcc
                            best_df.at[drug, 'mccvar'] = np.var(addmcc)
                            best_df.at[drug, 'precision'] = meanprecision
                            best_df.at[drug, 'precisionvar'] = np.var(addprecision)
                            best_df.at[drug, 'recall'] = meanrecall
                            best_df.at[drug, 'recallvar'] = np.var(addrecall)
                        print('pretrain mean auc:', sum(addauc) / len(addauc))
                        with open(test_auc_log_name, 'w') as f:
                            for item in test_auc_list:
                                f.write(str(item) + '\n')
    result_path = os.path.join(outfolder, resultname)
    best_df.to_csv(result_path, index=True)                   
    return all_metrics

def main_train_classifier(i):
    pretrain_model = os.path.join('result/pretrain','pretrain'+str(i))
    outfolder = os.path.join('result', 'prototypical')
    outname = f'p_result{i}.csv'
    safemakedirs(outfolder)
    drugmodel_pth = os.path.join('result', 'drug_encoder.pth')
    for dataset in ['PDTC', 'TCGA']:
        if dataset == 'PDTC':
            pdtc_drug_file = pd.read_csv(os.path.join('data', 'pdtc_gdsc_drug_mapping.csv'), index_col=0)
            drug_list = pdtc_drug_file.index.tolist()
            drug_smiles = pdtc_drug_file['smiles'].tolist()
            step1_metrics = step_1_finetune(pretrain_model, drug_list, drug_smiles, dataset, outfolder, outname, drugmodel_pth)
            classifier_file_path = './time_txt/pdtc'
            safemakedirs(classifier_file_path)
            classifier_file = './time_txt/pdtc/PDTC_mask_time'+str(i)+'.txt'
            with open(classifier_file, "w") as file:
                file.write(json.dumps(step1_metrics))
        elif  dataset == 'TCGA':
            drug_list = ['cis', 'sor', 'tem', 'gem', 'fu']
            drug_smiles = ['N.N.Cl[Pt]Cl', 'CNC(=O)C1=NC=CC(=C1)OC2=CC=C(C=C2)NC(=O)NC3=CC(=C(C=C3)Cl)C(F)(F)F',
                           'CN1C(=O)N2C=NC(=C2N=N1)C(=O)N', 'C1=CN(C(=O)N=C1N)C2C(C(C(O2)CO)O)(F)F', 'C1=C(C(=O)NC(=O)N1)F']
            step1_metrics = step_1_finetune(pretrain_model, drug_list, drug_smiles, dataset, outfolder, outname, drugmodel_pth)
            classifier_file_path = './time_txt/tcga'
            safemakedirs(classifier_file_path)
            classifier_file = './time_txt/tcga/tcga_mask_time'+str(i)+'.txt'
            with open(classifier_file, "w") as file:
                file.write(json.dumps(step1_metrics))

if __name__ == '__main__':  
    for i in range(10):
        parser = argparse.ArgumentParser('prototypical')
        parser.add_argument('--dataset', dest='dataset', default='TCGA', choices=['TCGA', 'PDTC', 'other'])
        parser.add_argument('--data', dest='data', type=str, default=None, help='data folder, if you use your own dataset(dataset == other) , this folder contains folders,'
                                                                                'that the folders contain sourcedata.csv sourcelabel.csv targetdata.csv targetlabel.csv')
        parser.add_argument('--drug', dest='drug', type=str, default=None, help='two col df, one contains drugnames, one contains smiles')
        parser.add_argument('--drugpth', dest='drugpth', type=str, default='./result/drug_encoder.pth', help='drug model path')
        parser.add_argument('--pretrain_model', dest='pretrain_model', type=str, default=f'./result/pretrain/pretrain{i}', help='pretrain model folder')
        parser.add_argument('--outfolder', dest='outfolder', type=str, default='./result/prototypical', help='folder to save result')
        parser.add_argument('--outname', dest='outname', type=str, default=f'p_result{i}.csv', help='result .csv file')
        args = parser.parse_args()
        safemakedirs(args.outfolder)
        drugmodel_pth = torch.load(args.drugpth)
        if args.dataset in ['TCGA', 'PDTC']:
            if args.dataset == 'PDTC':
                pdtc_drug_file = pd.read_csv(os.path.join('data', 'pdtc_gdsc_drug_mapping.csv'), index_col=0)
                drug_list = pdtc_drug_file.index.tolist()
                drug_smiles = pdtc_drug_file['smiles'].tolist()
                step1_metrics = step_1_finetune(args.pretrain_model, drug_list, drug_smiles, args.dataset, args.outfolder, 'PDTC_'+args.outname, drugmodel_pth)
                classifier_file_path = './time_txt/pdtc'
                safemakedirs(classifier_file_path)
                classifier_file = './time_txt/pdtc/PDTC_mask_time'+str(i)+'.txt'
                with open(classifier_file, "w") as file:
                    file.write(json.dumps(step1_metrics))
            elif args.dataset == 'TCGA':
                drug_list = ['cis', 'sor', 'tem', 'gem', 'fu']
                drug_smiles = ['N.N.Cl[Pt]Cl', 'CNC(=O)C1=NC=CC(=C1)OC2=CC=C(C=C2)NC(=O)NC3=CC(=C(C=C3)Cl)C(F)(F)F',
                            'CN1C(=O)N2C=NC(=C2N=N1)C(=O)N', 'C1=CN(C(=O)N=C1N)C2C(C(C(O2)CO)O)(F)F', 'C1=C(C(=O)NC(=O)N1)F']
                step1_metrics = step_1_finetune(args.pretrain_model, drug_list, drug_smiles, args.dataset, args.outfolder, 'TCGA_'+args.outname, drugmodel_pth)
                classifier_file_path = './time_txt/tcga'
                safemakedirs(classifier_file_path)
                classifier_file = './time_txt/tcga/tcga_mask_time'+str(i)+'.txt'
                with open(classifier_file, "w") as file:
                    file.write(json.dumps(step1_metrics))
        elif args.dataset == 'other':
            drug_file = pd.read_csv(args.drug, index_col=0)
            drug_list = drug_file.index.tolist()
            drug_smiles = drug_file['smiles'].tolist()
            step1_metrics = step_1_finetune(args.pretrain_model, drug_list, drug_smiles, args.dataset, args.outfolder, 'other_'+args.outname, drugmodel_pth, otherfolder=args.data)
            classifier_file_path = 'other_time'+str(i)+'.txt'
            with open(classifier_file, "w") as file:
                file.write(json.dumps(step1_metrics))