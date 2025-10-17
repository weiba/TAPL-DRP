import os
import torch
import math
import copy
import itertools
import argparse
import pandas as pd
import numpy as np
import torch.nn as nn
import torch.autograd as autograd
from data import *
from copy import deepcopy
from itertools import chain
from collections import defaultdict
from tools.dataprocess import *
from torch_geometric import data as DATA
from tools.model import *
from drugmodels.ginconv import GINConvNet
from sklearn.metrics import accuracy_score, f1_score, auc, precision_recall_curve, average_precision_score, roc_auc_score
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.manifold import TSNE

device = 'cuda' if torch.cuda.is_available() else 'cpu'

def safemakedirs(folder):
    if not os.path.exists(folder):
        os.makedirs(folder)

def ortho_loss(shared_z, private_z):
    s_l2_norm = torch.norm(shared_z, p=2, dim=1, keepdim=True).detach()
    s_l2 = shared_z.div(s_l2_norm.expand_as(shared_z) + 1e-6)
    p_l2_norm = torch.norm(private_z, p=2, dim=1, keepdim=True).detach()
    p_l2 = private_z.div(p_l2_norm.expand_as(private_z) + 1e-6)
    ortho_loss = torch.mean((s_l2.t().mm(p_l2)).pow(2))
    return ortho_loss

def compute_gradient_penalty(critic, real_samples, fake_samples, device):
    alpha = torch.rand((real_samples.shape[0], 1)).to(device)
    interpolates = (alpha * real_samples + ((1 - alpha) * fake_samples)).requires_grad_(True)
    critic_interpolates = critic(interpolates)
    fakes = torch.ones((real_samples.shape[0], 1)).to(device)
    gradients = autograd.grad(
        outputs=critic_interpolates,
        inputs=interpolates,
        grad_outputs=fakes,
        create_graph=True,
        retain_graph=True,
        only_inputs=True,
    )[0]
    gradients = gradients.view(gradients.size(0), -1)
    gradient_penalty = ((gradients.norm(2, dim=1) - 1) ** 2).mean()
    return gradient_penalty

def train_discrim(s_batch, t_batch, shared_encoder, sencoder, tencoder, discrim, optimizer, scheduler):
    loss_log = defaultdict(float)
    shared_encoder.zero_grad()
    sencoder.zero_grad()
    tencoder.zero_grad()
    discrim.zero_grad()
    sencoder.eval()
    tencoder.eval()
    shared_encoder.eval()
    discrim.train()
    optimizer.zero_grad()

    _, pzs, _, _ = sencoder(s_batch)
    _, pzt, _, _ = tencoder(t_batch)
    _, zs, _, _ = shared_encoder(s_batch)
    _, zt, _, _ = shared_encoder(t_batch)
    s = torch.cat((zs, pzs), dim=1)
    t = torch.cat((zt, pzt), dim=1)
    d_loss = torch.mean(t) - torch.mean(s)
    g_p = compute_gradient_penalty(
        critic=discrim,
        real_samples=s,
        fake_samples=t,
        device=device
    )
    loss_log.update({"discrim_loss": d_loss, "g_p": g_p})
    d_loss = d_loss + 10 * g_p
    d_loss.backward(retain_graph=True)  
    optimizer.step()
    scheduler.step()
    discrim.eval()
    return loss_log

def train_d_ae(s_batch, t_batch, s_labels, t_labels, shared_encoder, sencoder, tencoder, discrim, optimizer, scheduler, proto_manager, unique_labels, epoch, device='cuda', samples_per_cls=None):
    loss_log = defaultdict(float)
    shared_encoder.zero_grad()
    sencoder.zero_grad()
    tencoder.zero_grad()
    discrim.zero_grad()
    sencoder.train()
    tencoder.train()
    shared_encoder.train()
    discrim.eval()
    optimizer.zero_grad()

    pccle_re_x, pccle_z, pccle_mu, pccle_sigma = sencoder(s_batch)
    ptcga_re_x, ptcga_z, ptcga_mu, ptcga_sigma = tencoder(t_batch)
    ccle_re_x, ccle_z, ccle_mu, ccle_sigma = shared_encoder(s_batch)
    tcga_re_x, tcga_z, tcga_mu, tcga_sigma = shared_encoder(t_batch)

    pccle_vae_loss = vaeloss(pccle_mu, pccle_sigma, pccle_re_x, s_batch)
    ptcga_vae_loss = vaeloss(ptcga_mu, ptcga_sigma, ptcga_re_x, t_batch)
    ccle_vae_loss = vaeloss(ccle_mu, ccle_sigma, ccle_re_x, s_batch)
    tcga_vae_loss = vaeloss(tcga_mu, tcga_sigma, tcga_re_x, t_batch)

    pvae_loss = pccle_vae_loss + ptcga_vae_loss
    vae_loss = ccle_vae_loss + tcga_vae_loss
    o_loss = ortho_loss(ccle_z, pccle_z) + ortho_loss(tcga_z, ptcga_z)

    g_loss = -torch.mean(discrim(torch.cat((tcga_z, ptcga_z), dim=1)))
    z = torch.cat((ccle_z, tcga_z), dim=0)
    labels = torch.cat((s_labels, t_labels), dim=0)
    batch_size = z.size(0)
    if batch_size == 0:
        print("Warning: Batch size is 0, skipping InfoNCE loss")
        nce_loss = torch.tensor(0.0).to(device)
    else:
        max_label = labels.max().item()
        num_classes = max(max_label + 1, len(unique_labels))
        nce_loss = info_nce_loss(z, labels, temperature=0.5, device=device, num_classes=num_classes)

    with torch.no_grad():
        proto_manager.update_batch(ccle_z, s_labels, domain='ccle', epoch=epoch)
        proto_manager.update_batch(tcga_z, t_labels, domain='tcga', epoch=epoch)
    proto_loss = proto_manager.compute_cross_domain_loss()

    no_of_classes = len(unique_labels)
    ccle_logits = ccle_z
    tcga_logits = tcga_z
    if ccle_logits.size(0) > 0 and samples_per_cls is not None:
        linear_layer = nn.Linear(32, no_of_classes).to(device)
        ccle_logits = linear_layer(ccle_z)
        tcga_logits = linear_layer(tcga_z)
        ccle_cb_loss = cb_loss(s_labels, ccle_logits, samples_per_cls['ccle'], unique_labels)
        tcga_cb_loss = cb_loss(t_labels, tcga_logits, samples_per_cls['tcga'], unique_labels)
        cb_loss_value = (ccle_cb_loss + tcga_cb_loss) / 2
    else:
        cb_loss_value = torch.tensor(0.0).to(device)
        print("Warning: Invalid batch for CB loss, setting to 0.")

    loss = o_loss + vae_loss + pvae_loss + g_loss + 1.0 * proto_loss + 0.8 * nce_loss + 0.5 * cb_loss_value
    loss_log.update({
        "ortho_loss": o_loss.item(),
        "pvae_loss": pvae_loss.item(),
        "vae_loss": vae_loss.item(),
        "g_loss": g_loss.item(),
        "proto_loss": proto_loss.item(),
        "nce_loss": nce_loss.item(),
        "cb_loss": cb_loss_value.item()
    })
    
    loss.backward()
    optimizer.step()
    scheduler.step()
    return loss_log

def pretrain(sourcedata, targetdata, param, parent_folder, batch_size):
    print("start pretrain")
    set_dir_name = 'pt_epochs_' + str(param['pretrain_num_epochs']) + \
                   ',t_epochs_' + str(param['train_num_epochs']) + \
                   ',Ptlr_' + str(param['pretrain_learning_rate']) + \
                   ',tlr' + str(param['gan_learning_rate'])
    pretrain_dir = os.path.join(parent_folder, set_dir_name)
    safemakedirs(pretrain_dir)

    trainloss_logfile = os.path.join(pretrain_dir, "pretrain_losslog.txt")
    evalloss_logfile = os.path.join(pretrain_dir, "pretrain_eval_losslog.txt")

    trainloss_logdict = defaultdict(float)
    evalloss_logdict = defaultdict(float)

    sourcetrainloader, sourcetest = sourcedata
    targettrainloader, targettest = targetdata
    unique_labels = param['unique_labels']

    samples_per_cls = {
        'ccle': compute_samples_per_cls(sourcetrainloader, unique_labels),
        'tcga': compute_samples_per_cls(targettrainloader, unique_labels)
    }

    shared_vae = VAE(input_size=1426, output_size=1426, latent_size=32, hidden_size=128).to(device)
    source_private_vae = VAE(input_size=1426, output_size=1426, latent_size=32, hidden_size=128).to(device)
    target_private_vae = VAE(input_size=1426, output_size=1426, latent_size=32, hidden_size=128).to(device)

    proto_manager = PrototypeManager(
        unique_labels=unique_labels, 
        latent_dim=32, 
        alpha=0.9, 
        source_loader=sourcetrainloader, 
        target_loader=targettrainloader, 
        shared_vae=shared_vae
    )

    if not os.path.exists(os.path.join(pretrain_dir, "shared_vae.pth")):
        source_dict = source_private_vae.state_dict()
        shared_dict = shared_vae.state_dict()
        target_dict = target_private_vae.state_dict()
        pretrain_epochs = param['pretrain_num_epochs']
        learning_rate = param['pretrain_learning_rate']
        tolerance = 0
        max_tolerance = 50
        min_loss = float('inf')
        models = [shared_vae, source_private_vae, target_private_vae]
        models_parameters = [
            shared_vae.parameters(),
            source_private_vae.parameters(),
            target_private_vae.parameters()
        ]
        optimizer = torch.optim.Adam(chain(*models_parameters), lr=learning_rate)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, pretrain_epochs)
        for epoch in range(pretrain_epochs):
            if epoch % 20 == 0:
                print("pretrain epoch:", epoch)
            if epoch % 5 == 0:
                proto_manager.update_full(sourcetrainloader, shared_vae, domain='ccle')
                proto_manager.update_full(targettrainloader, shared_vae, domain='tcga')
                print(f"Epoch {epoch}: Prototypes updated with full dataset")
            
            train_epoch_oloss = 0
            train_epoch_vaeloss = 0
            train_epoch_pvaeloss = 0
            train_epoch_proto_loss = 0
            for model in models:
                model.train()
            for i, ((ccledata, ccle_labels), (tcgadata, tcga_labels)) in enumerate(zip(sourcetrainloader, targettrainloader)):
                ccledata = ccledata.to(device)
                tcgadata = tcgadata.to(device)
                ccle_labels = ccle_labels.to(device)
                tcga_labels = tcga_labels.to(device)
                optimizer.zero_grad()

                pccle_re_x, pccle_z, pccle_mu, pccle_sigma = source_private_vae(ccledata)
                pccle_vae_loss = vaeloss(pccle_mu, pccle_sigma, pccle_re_x, ccledata)
                ptcga_re_x, ptcga_z, ptcga_mu, ptcga_sigma = target_private_vae(tcgadata)
                ptcga_vae_loss = vaeloss(ptcga_mu, ptcga_sigma, ptcga_re_x, tcgadata)

                ccle_re_x, ccle_z, ccle_mu, ccle_sigma = shared_vae(ccledata)
                ccle_vae_loss = vaeloss(ccle_mu, ccle_sigma, ccle_re_x, ccledata)
                tcga_re_x, tcga_z, tcga_mu, tcga_sigma = shared_vae(tcgadata)
                tcga_vae_loss = vaeloss(tcga_mu, tcga_sigma, tcga_re_x, tcgadata)

                p_vae_loss = pccle_vae_loss + ptcga_vae_loss
                vae_loss = ccle_vae_loss + tcga_vae_loss
                o_loss = ortho_loss(ccle_z, pccle_z) + ortho_loss(tcga_z, ptcga_z)

                with torch.no_grad():
                    proto_manager.update_batch(ccle_z, ccle_labels, domain='ccle')
                    proto_manager.update_batch(tcga_z, tcga_labels, domain='tcga')
                proto_loss = proto_manager.compute_cross_domain_loss(ccle_labels, tcga_labels)

                loss = o_loss + vae_loss + p_vae_loss + 0.5 * proto_loss
                loss.backward(retain_graph=True)
                optimizer.step()
                scheduler.step()
                train_epoch_oloss += o_loss.item()
                train_epoch_vaeloss += vae_loss.item()
                train_epoch_pvaeloss += p_vae_loss.item()
                train_epoch_proto_loss += proto_loss.item()

            train_epoch_oloss = train_epoch_oloss / (i + 1)
            train_epoch_pvaeloss = train_epoch_pvaeloss / (i + 1)
            train_epoch_vaeloss = train_epoch_vaeloss / (i + 1)
            train_epoch_proto_loss = train_epoch_proto_loss / (i + 1)
            trainloss_logdict.update({
                "epoch": epoch,
                "ortholoss": train_epoch_oloss,
                "pVAE_loss": train_epoch_pvaeloss,
                "VAE_loss": train_epoch_vaeloss,
                "proto_loss": train_epoch_proto_loss
            })
            append_file(trainloss_logfile, trainloss_logdict)

            for model in models:
                model.eval()
            with torch.no_grad():
                ccle_test_loader = DataLoader(sourcetest, batch_size=batch_size, shuffle=False)
                tcga_test_loader = DataLoader(targettest, batch_size=batch_size, shuffle=False)
                ccle_eval_vae_loss = 0.0
                ccle_eval_pvae_loss = 0.0
                ccle_eval_oloss = 0.0
                ccle_eval_proto_loss = 0.0
                ccle_batches = 0
                for ccle_data, ccle_labels in ccle_test_loader:
                    ccle_data = ccle_data.to(device)
                    ccle_labels = ccle_labels.to(device)
                    pccle_re_x, pccle_z, pccle_mu, pccle_sigma = source_private_vae(ccle_data)
                    pccle_vae_loss = vaeloss(pccle_mu, pccle_sigma, pccle_re_x, ccle_data)
                    ccle_re_x, ccle_z, ccle_mu, ccle_sigma = shared_vae(ccle_data)
                    ccle_vae_loss = vaeloss(ccle_mu, ccle_sigma, ccle_re_x, ccle_data)
                    ccle_oloss = ortho_loss(ccle_z, pccle_z)
                    proto_manager.update_batch(ccle_z, ccle_labels, domain='ccle')
                    ccle_eval_vae_loss += ccle_vae_loss.item()
                    ccle_eval_pvae_loss += pccle_vae_loss.item()
                    ccle_eval_oloss += ccle_oloss.item()
                    ccle_eval_proto_loss += proto_manager.compute_cross_domain_loss(ccle_labels, None).item()
                    ccle_batches += 1

                tcga_eval_vae_loss = 0.0
                tcga_eval_pvae_loss = 0.0
                tcga_eval_oloss = 0.0
                tcga_eval_proto_loss = 0.0
                tcga_batches = 0
                for tcga_data, tcga_labels in tcga_test_loader:
                    tcga_data = tcga_data.to(device)
                    tcga_labels = tcga_labels.to(device)
                    ptcga_re_x, ptcga_z, ptcga_mu, ptcga_sigma = target_private_vae(tcga_data)
                    ptcga_vae_loss = vaeloss(ptcga_mu, ptcga_sigma, ptcga_re_x, tcga_data)
                    tcga_re_x, tcga_z, tcga_mu, tcga_sigma = shared_vae(tcga_data)
                    tcga_vae_loss = vaeloss(tcga_mu, tcga_sigma, tcga_re_x, tcga_data)
                    tcga_oloss = ortho_loss(tcga_z, ptcga_z)
                    proto_manager.update_batch(tcga_z, tcga_labels, domain='tcga')
                    tcga_eval_vae_loss += tcga_vae_loss.item()
                    tcga_eval_pvae_loss += ptcga_vae_loss.item()
                    tcga_eval_oloss += tcga_oloss.item()
                    tcga_eval_proto_loss += proto_manager.compute_cross_domain_loss(None, tcga_labels).item()
                    tcga_batches += 1

                eval_vae_loss = (ccle_eval_vae_loss + tcga_eval_vae_loss) / (ccle_batches + tcga_batches)
                eval_pvae_loss = (ccle_eval_pvae_loss + tcga_eval_pvae_loss) / (ccle_batches + tcga_batches)
                eval_oloss = (ccle_eval_oloss + tcga_eval_oloss) / (ccle_batches + tcga_batches)
                eval_proto_loss = (ccle_eval_proto_loss + tcga_eval_proto_loss) / (ccle_batches + tcga_batches)
                evalloss_logdict.update({
                    "epoch": epoch,
                    "ortholoss": eval_oloss,
                    "pVAE_loss": eval_pvae_loss,
                    "VAE_loss": eval_vae_loss,
                    "proto_loss": eval_proto_loss
                })
                append_file(evalloss_logfile, evalloss_logdict)
                evalloss = eval_oloss + eval_pvae_loss + eval_vae_loss + 0.5 * eval_proto_loss
                if evalloss < min_loss:
                    min_loss = evalloss
                    tolerance = 0
                    source_dict = source_private_vae.state_dict()
                    target_dict = target_private_vae.state_dict()
                    shared_dict = shared_vae.state_dict()
                else:
                    tolerance += 1
                if tolerance >= max_tolerance:
                    print("pretrain early stop")
                    break
        torch.save(shared_dict, os.path.join(pretrain_dir, "shared_vae.pth"))
        torch.save(source_dict, os.path.join(pretrain_dir, "source_vae.pth"))
        torch.save(target_dict, os.path.join(pretrain_dir, "target_vae.pth"))
        
        proto_manager.update_full(sourcetrainloader, shared_vae, domain='ccle')
        proto_manager.update_full(targettrainloader, shared_vae, domain='tcga')
    else:
        shared_dict = torch.load(os.path.join(pretrain_dir, "shared_vae.pth"))
        source_dict = torch.load(os.path.join(pretrain_dir, "source_vae.pth"))
        target_dict = torch.load(os.path.join(pretrain_dir, "target_vae.pth"))

    if os.path.exists(os.path.join(pretrain_dir, 'after_traingan_shared_vae.pth')):
        print("after train gan model exists")
    else:
        print("start gan train")
        gan_epoch = param['train_num_epochs']
        gan_lr = param['gan_learning_rate']
        shared_vae.load_state_dict(shared_dict)
        source_private_vae.load_state_dict(source_dict)
        target_private_vae.load_state_dict(target_dict)
        d_ae_parameters = [
            shared_vae.parameters(),
            source_private_vae.parameters(),
            target_private_vae.parameters()
        ]
        discrim = Discriminator(input_dim=32 + 32).to(device)
        discrim_optimizer = torch.optim.RMSprop(discrim.parameters(), lr=gan_lr)
        discrim_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(discrim_optimizer, gan_epoch)
        d_ae_optimizer = torch.optim.RMSprop(chain(*d_ae_parameters), lr=gan_lr)
        d_ae_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(d_ae_optimizer, gan_epoch)
        dloss_logfile = os.path.join(pretrain_dir, "d_losslog.txt")
        genloss_logfile = os.path.join(pretrain_dir, "g_losslog.txt")
        max_gan_tolerance = 20
        gan_tolerance = 0
        shared_vae_aftergan_dict = shared_vae.state_dict()
        min_loss = float('inf')
        for epoch in range(gan_epoch):
            if epoch % 5 == 0:
                proto_manager.update_full(sourcetrainloader, shared_vae, domain='ccle')
                proto_manager.update_full(targettrainloader, shared_vae, domain='tcga')
                print(f"GAN Epoch {epoch}: Prototypes updated with full dataset")
            
            temp_loss = 0
            dloss_list = []
            genloss_list = []
            if epoch % 10 == 0:
                print(f'confounder wgan training epoch {epoch}')
            for i, ((ccledata, ccle_labels), (tcgadata, tcga_labels)) in enumerate(zip(sourcetrainloader, targettrainloader)):
                ccledata = ccledata.to(device)
                tcgadata = tcgadata.to(device)
                ccle_labels = ccle_labels.to(device)
                tcga_labels = tcga_labels.to(device)
                
                discrim_optimizer.zero_grad()
                dlosslog = train_discrim(
                    s_batch=ccledata,
                    t_batch=tcgadata,
                    shared_encoder=shared_vae,
                    sencoder=source_private_vae,
                    tencoder=target_private_vae,
                    discrim=discrim,
                    optimizer=discrim_optimizer,
                    scheduler=discrim_scheduler
                )
                dloss_list.append(dlosslog)
                
                if (i + 1) % 5 == 0:
                    d_ae_optimizer.zero_grad()
                    genlosslog = train_d_ae(
                        s_batch=ccledata,
                        t_batch=tcgadata,
                        s_labels=ccle_labels,
                        t_labels=tcga_labels,
                        shared_encoder=shared_vae,
                        sencoder=source_private_vae,
                        tencoder=target_private_vae,
                        discrim=discrim,
                        optimizer=d_ae_optimizer,
                        scheduler=d_ae_scheduler,
                        proto_manager=proto_manager,
                        unique_labels=unique_labels,
                        epoch=epoch, 
                        samples_per_cls=samples_per_cls
                    )
                    genloss_list.append(genlosslog)
                
            dloss_sum = defaultdict(float)
            dloss_mean = defaultdict(float)
            for metric_dict in dloss_list:
                for metric, value in metric_dict.items():
                    dloss_sum[metric] += value.item()
            num_dicts = len(dloss_list)
            for metric, total_value in dloss_sum.items():
                average_value = total_value / num_dicts if num_dicts > 0 else 0
                dloss_mean[metric] = average_value
            genloss_sum = defaultdict(float)
            genloss_mean = defaultdict(float)
            for metric_dict in genloss_list:
                for metric, value in metric_dict.items():
                    genloss_sum[metric] += value
            num_dicts = len(genloss_list)
            for metric, total_value in genloss_sum.items():
                average_value = total_value / num_dicts if num_dicts > 0 else 0
                genloss_mean[metric] = average_value
            append_file(dloss_logfile, dloss_mean)
            append_file(genloss_logfile, genloss_mean)
            for key in dloss_mean:
                temp_loss += dloss_mean[key]
            for key in genloss_mean:
                temp_loss += genloss_mean[key]
            if min_loss > temp_loss:
                gan_tolerance = 0
                shared_vae_aftergan_dict = shared_vae.state_dict()
                min_loss = temp_loss
            else:
                gan_tolerance += 1
            if gan_tolerance >= max_gan_tolerance:
                print("train gan early stop in epoch:", epoch)
                break
        torch.save(shared_vae_aftergan_dict, os.path.join(pretrain_dir, "after_traingan_shared_vae.pth"))
        
        proto_manager.update_full(sourcetrainloader, shared_vae, domain='ccle')
        proto_manager.update_full(targettrainloader, shared_vae, domain='tcga')

def main_pretrain(i):
    params_grid = {
        "pretrain_num_epochs": [0, 100, 300],
        'pretrain_learning_rate': [0.001],
        'gan_learning_rate': [0.001],
        "train_num_epochs": [100, 200, 300, 500, 750, 1000, 1500, 2000, 2500, 3000]
    }
    keys, values = zip(*params_grid.items())
    update_params_dict_list = [dict(zip(keys, v)) for v in itertools.product(*values)]
    sourcepretrain, targetpretrain, unique_labels, batch_size = pretrain_data()
    pretrain_path = os.path.join('result/pretrain', 'pretrain' + str(i))
    safemakedirs(pretrain_path)
    for param_dict in update_params_dict_list:
        param_dict['unique_labels'] = unique_labels
        pretrain(sourcepretrain, targetpretrain, param=param_dict, parent_folder=pretrain_path, batch_size=batch_size)

if __name__ == '__main__':
    for i in range(10):
        parser = argparse.ArgumentParser('pretrain')
        parser.add_argument('--outfolder', dest='outfolder', default=f'./result/pretrain_测试pretrain代码/pretrain{i}', type=str, help='choose the output folder')
        parser.add_argument('--source', dest='source', default=None, type=str, help='.csv file address for the source')
        parser.add_argument('--target', dest='target', default=None, type=str, help='.csv file address for the target')
        args = parser.parse_args()
        params_grid = {
            "pretrain_num_epochs": [0, 100, 300],
            'pretrain_learning_rate': [0.001],
            'gan_learning_rate': [0.001],
            "train_num_epochs": [100, 200, 300, 500, 750, 1000, 1500, 2000, 2500, 3000]
        }
        keys, values = zip(*params_grid.items())
        update_params_dict_list = [dict(zip(keys, v)) for v in itertools.product(*values)]
        safemakedirs(args.outfolder)
        if args.source and args.target:
            sourcepretrain = pd.read_csv(args.source, index_col=0, header=0)
            targetpretrain = pd.read_csv(args.target, index_col=0, header=0)
            sourcepretrain = pretrain_loader(sourcepretrain)
            targetpretrain = pretrain_loader(targetpretrain)
            # Placeholder: Set unique_labels to single class for custom data
            # In practice, this should be determined from your actual data labels
            unique_labels = list(range(1))  
            # Placeholder: Set batch size for training
            # Adjust this value based on your GPU memory and data size
            batch_size = 64 
            for param_dict in update_params_dict_list:
                param_dict['unique_labels'] = unique_labels
                pretrain(sourcepretrain, targetpretrain, param=param_dict, parent_folder=args.outfolder, batch_size=batch_size)
        elif args.source and args.target == None:
            sourcepretrain = pd.read_csv(args.source, index_col=0, header=0)
            sourcepretrain = pretrain_loader(sourcepretrain)
            _, targetpretrain, unique_labels, batch_size = pretrain_data()
            for param_dict in update_params_dict_list:
                param_dict['unique_labels'] = unique_labels
                pretrain(sourcepretrain, targetpretrain, param=param_dict, parent_folder=args.outfolder, batch_size=batch_size)
        elif args.source == None and args.target:
            sourcepretrain, _, unique_labels, batch_size = pretrain_data()
            targetpretrain = pd.read_csv(args.target, index_col=0, header=0)
            targetpretrain = pretrain_loader(targetpretrain)
            for param_dict in update_params_dict_list:
                param_dict['unique_labels'] = unique_labels
                pretrain(sourcepretrain, targetpretrain, param=param_dict, parent_folder=args.outfolder, batch_size=batch_size)
        else:
            sourcepretrain, targetpretrain, unique_labels, batch_size = pretrain_data()
            for param_dict in update_params_dict_list:
                param_dict['unique_labels'] = unique_labels
                pretrain(sourcepretrain, targetpretrain, param=param_dict, parent_folder=args.outfolder, batch_size=batch_size)