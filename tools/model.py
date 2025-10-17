import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from tools.dataprocess import *
from torch_geometric import data as DATA
from torch_geometric.nn import GCNConv, GINConv, GATConv, ChebConv, GAE, global_mean_pool, global_max_pool

device = 'cuda' if torch.cuda.is_available() else 'cpu'

class VariationalGCNEncoder(torch.nn.Module):
    def __init__(self, in_channels, out_channels):
        super(VariationalGCNEncoder, self).__init__()
        self.conv1 = ChebConv(in_channels, 2 * out_channels, K=2)
        self.bn1 = nn.BatchNorm1d(2 * out_channels)
        self.conv_mu = ChebConv(2 * out_channels, out_channels, K=2)
        self.conv_logstd = ChebConv(2 * out_channels, out_channels, K=2)
        self.relu = nn.ReLU()
    def forward(self, data: DATA.data, batch=None):
        x, edge_index = data.x, data.edge_index
        x = self.conv1(x, edge_index)
        x = self.bn1(x)
        x = self.relu(x)
        mu, logstd = self.conv_mu(x, edge_index), self.conv_logstd(x, edge_index)
        return mu, logstd


class gEncoder(nn.Module):
    def __init__(self, input_dim:int, output_dim:int, dropout=0.2):
        super(gEncoder, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, input_dim//2),
            nn.BatchNorm1d(input_dim//2),
            nn.ReLU(),
            nn.Dropout(p=dropout),
            nn.Linear(input_dim//2, output_dim)
        )
    def forward(self, x):
        return self.net(x)


class gDecoder(nn.Module):
    def __init__(self, recon_dim:int, emb_dim:int, dropout=0.2):
        super(gDecoder, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(emb_dim, emb_dim*2),
            nn.BatchNorm1d(emb_dim*2),
            nn.ReLU(),
            nn.Linear(emb_dim*2, recon_dim)
        )
    def forward(self, x):
        return self.net(x)


class GraphEncoder(nn.Module):
    def __init__(self, input_dim: int, output_dim: int, dropout=0.2):
        super(GraphEncoder, self).__init__()
        self.conv1 = ChebConv(in_channels=input_dim, out_channels=128, K=2)
        self.bn1 = nn.BatchNorm1d(128)
        self.conv2 = ChebConv(in_channels=128, out_channels=128, K=2)
        self.bn2 = nn.BatchNorm1d(128)
        self.fc_g1 = torch.nn.Linear(128, 64)
        self.bn3 = nn.BatchNorm1d(64)
        self.fc_g2 = torch.nn.Linear(64, output_dim)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout)

    def forward(self, data: DATA.data, batch=None):
        x, edge_index = data.x, data.edge_index
        x = self.conv1(x, edge_index)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.conv2(x, edge_index)
        x = self.bn2(x)
        x = self.relu(x)
        x = self.fc_g1(x)
        x = self.bn3(x)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.fc_g2(x)
        x_mean = global_mean_pool(x, batch=batch)
        return x, x_mean


class GraphDecoder(nn.Module):
    def __init__(self, recon_dim: int, emb_dim: int, dropout=0.2):
        super(GraphDecoder, self).__init__()
        self.fc_g1 = torch.nn.Linear(emb_dim, 1024)
        self.fc_g2 = torch.nn.Linear(1024, recon_dim)
        self.conv1 = ChebConv(recon_dim, recon_dim, K=2)
        self.relu = nn.ReLU()

    def forward(self, x: torch.Tensor, edge_index:torch.Tensor):
        x = self.fc_g1(x)
        x = self.relu(x)
        x = self.fc_g2(x)
        x = self.relu(x)
        x = self.conv1(x, edge_index)
        return x


class Edgeindexdecoder(nn.Module):
    def __init__(self, input_dim:int):
        super(Edgeindexdecoder, self).__init__()
        self.fc1 = torch.nn.Linear(input_dim, input_dim)
        self.fc2 = torch.nn.Linear(input_dim, input_dim)
        self.relu = nn.ReLU()
        self.drop = nn.Dropout(p=0.2)

    def forward(self, edge_index:torch.tensor):
        edge_index = self.fc1(edge_index)
        edge_index = self.relu(edge_index)
        edge_index = self.drop(edge_index)
        edge_index = self.fc2(edge_index)
        return edge_index

class Classify(nn.Module):
    def __init__(self, input_dim):
        super(Classify, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 10),
            nn.BatchNorm1d(10),
            nn.ReLU(),
            nn.Linear(10, 1)
        )

    def forward(self, x):
        return (self.net(x)).view(-1)

class Discriminator(nn.Module):
    def __init__(self, input_dim):
        super(Discriminator, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, input_dim // 2),
            nn.ReLU(),
            nn.Linear(input_dim // 2, 1),
        )

    def forward(self, x):
        return self.net(x)

class VAE_Encoder(torch.nn.Module):
    def __init__(self, input_size, hidden_size, latent_size):
        super(VAE_Encoder, self).__init__()
        self.linear = torch.nn.Linear(input_size, hidden_size)
        self.mu = torch.nn.Linear(hidden_size, latent_size)
        self.sigma = torch.nn.Linear(hidden_size, latent_size)
    def forward(self, x):# x: bs,input_size
        x = F.relu(self.linear(x)) #-> bs,hidden_size
        mu = self.mu(x) #-> bs,latent_size
        sigma = self.sigma(x)#-> bs,latent_size
        return mu,sigma

class VAE_Decoder(torch.nn.Module):
    def __init__(self, latent_size, hidden_size, output_size):
        super(VAE_Decoder, self).__init__()
        self.linear1 = torch.nn.Linear(latent_size, hidden_size)
        self.linear2 = torch.nn.Linear(hidden_size, output_size)
    def forward(self, x): # x:bs,latent_size
        x = F.relu(self.linear1(x)) #->bs,hidden_size
        x = self.linear2(x)
        return x

class VAE(torch.nn.Module):
    def __init__(self, input_size, output_size, latent_size, hidden_size):
        super(VAE, self).__init__()
        self.encoder = VAE_Encoder(input_size, hidden_size, latent_size)
        self.decoder = VAE_Decoder(latent_size, hidden_size, output_size)

    def forward(self, x): #x: bs,input_size
        mu,sigma = self.encoder(x) #mu,sigma: bs,latent_size
        std = torch.exp(0.5 * sigma)
        eps = torch.randn_like(std)  #eps: bs,latent_size
        z = mu + eps*sigma  #z: bs,latent_size
        re_x = self.decoder(z)  # re_x: bs,output_size
        return re_x,z,mu,sigma

def vaeloss(mu, sigma, re_x, x, alpha=0.1):
    mseloss = torch.nn.MSELoss()
    recon_loss = mseloss(re_x, x)
    KLD = -0.5 * torch.sum(1 + sigma - mu.pow(2) - sigma.exp())
    loss = alpha*KLD+recon_loss
    return loss

class PrototypeManager:
    def __init__(self, unique_labels, latent_dim, alpha=0.8, source_loader=None, target_loader=None, shared_vae=None):
        self.unique_labels = unique_labels
        self.latent_dim = latent_dim
        self.initial_alpha = alpha  
        self.prototypes_ccle = {label: torch.zeros(latent_dim).to(device) for label in unique_labels}
        self.prototypes_tcga = {label: torch.zeros(latent_dim).to(device) for label in unique_labels}
        self.counts_ccle = {label: 0 for label in unique_labels}
        self.counts_tcga = {label: 0 for label in unique_labels}

        if source_loader is not None and target_loader is not None and shared_vae is not None:
            self.update_full(source_loader, shared_vae, domain='ccle')
            self.update_full(target_loader, shared_vae, domain='tcga')

    def update_batch(self, z, labels, domain='ccle', epoch=None):
        prototypes = self.prototypes_ccle if domain == 'ccle' else self.prototypes_tcga
        counts = self.counts_ccle if domain == 'ccle' else self.counts_tcga
        
        alpha = self.initial_alpha if epoch is None else max(0.5, self.initial_alpha - epoch * 0.002)
        # alpha = 0.9
        for label in set(labels.cpu().numpy()):
            mask = (labels == label)
            if mask.sum() > 0:
                mean_z = z[mask].mean(dim=0)
                if counts[label] == 0:
                    prototypes[label] = mean_z
                else:
                    prototypes[label] = alpha * prototypes[label] + (1 - alpha) * mean_z
                counts[label] += 1

    def update_full(self, data_loader, shared_vae, domain='ccle'):
        prototypes = self.prototypes_ccle if domain == 'ccle' else self.prototypes_tcga
        counts = self.counts_ccle if domain == 'ccle' else self.counts_tcga
        temp_prototypes = {label: torch.zeros(self.latent_dim).to(device) for label in self.unique_labels}
        temp_counts = {label: 0 for label in self.unique_labels}
        
        shared_vae.eval()
        with torch.no_grad():
            for batch, labels in data_loader:
                batch = batch.to(device)
                _, z, _, _ = shared_vae(batch)
                for label in set(labels.cpu().numpy()):
                    mask = (labels == label)
                    if mask.sum() > 0:
                        temp_prototypes[label] += z[mask].sum(dim=0)
                        temp_counts[label] += mask.sum().item()
        
        for label in self.unique_labels:
            if temp_counts[label] > 0:
                prototypes[label] = temp_prototypes[label] / temp_counts[label]
                counts[label] = temp_counts[label]

    def compute_cross_domain_loss(self, labels_ccle=None, labels_tcga=None):
        loss = 0.0
        valid_pairs = 0
        for label in self.unique_labels:
            ccle_proto = self.prototypes_ccle[label]
            tcga_proto = self.prototypes_tcga[label]
            if torch.any(ccle_proto != 0) and torch.any(tcga_proto != 0):
                l2_loss = torch.norm(ccle_proto - tcga_proto, p=2) ** 2
                similarity_loss = torch.log(1 + torch.exp(-torch.dot(ccle_proto, tcga_proto)))
                org_consistency = torch.norm(ccle_proto - tcga_proto, p=2) ** 2
                loss += l2_loss + 0.1 * similarity_loss + 0.5 * org_consistency
                valid_pairs += 1
        return loss / valid_pairs if valid_pairs > 0 else 0.0
    

def info_nce_loss(z, labels, temperature=0.5, device='cuda', num_classes=None):
    batch_size = z.size(0)
    if batch_size <= 1:  
        print("Warning: Batch size <= 1, skipping InfoNCE loss")
        return torch.tensor(0.0, requires_grad=True).to(device)
    
    z = F.normalize(z, dim=1)
    sim_matrix = torch.matmul(z, z.T) / temperature
    if torch.isnan(sim_matrix).any() or torch.isinf(sim_matrix).any():
        print("Warning: sim_matrix contains NaN or Inf, skipping InfoNCE loss")
        return torch.tensor(0.0, requires_grad=True).to(device)
    
    if num_classes is None:
        num_classes = len(torch.unique(labels))
    if num_classes <= 1:
        print("Warning: Only one class in batch, skipping InfoNCE loss")
        return torch.tensor(0.0, requires_grad=True).to(device)
    
    labels_one_hot = F.one_hot(labels, num_classes=num_classes).float().to(device)
    positive_mask = torch.matmul(labels_one_hot, labels_one_hot.T)
    
    logits_mask = 1 - torch.eye(batch_size).to(device)
    exp_logits = torch.exp(sim_matrix) * logits_mask
    log_prob = sim_matrix - torch.log(exp_logits.sum(1, keepdim=True) + 1e-8)
    
    mean_log_prob_pos = (positive_mask * log_prob).sum(1) / (positive_mask.sum(1) + 1e-8)
    loss = -mean_log_prob_pos.mean()
    
    if torch.isnan(loss).any() or torch.isinf(loss).any():
        print("Warning: info_nce_loss contains NaN or Inf, returning 0.0")
        return torch.tensor(0.0, requires_grad=True).to(device)
    
    return loss

def prototype_contrastive_loss(z, labels, prototypes, temperature=0.5, device='cuda'):
    batch_size = z.size(0)
    if batch_size <= 1:
        print("Warning: Batch size <= 1, skipping prototype contrastive loss")
        return torch.tensor(0.0, requires_grad=True).to(device)
    
    z = F.normalize(z, dim=1)
    proto_tensors = []
    proto_labels = []
    for label in prototypes.keys():
        if torch.any(prototypes[label] != 0):
            proto_tensors.append(F.normalize(prototypes[label], dim=0))
            proto_labels.append(label)
    
    if not proto_tensors:
        print("Warning: No valid prototypes, skipping prototype contrastive loss")
        return torch.tensor(0.0, requires_grad=True).to(device)
    
    proto_tensors = torch.stack(proto_tensors).to(device)
    sim_matrix = torch.matmul(z, proto_tensors.T) / temperature
    
    if torch.isnan(sim_matrix).any() or torch.isinf(sim_matrix).any():
        print("Warning: sim_matrix contains NaN or Inf, skipping prototype contrastive loss")
        return torch.tensor(0.0, requires_grad=True).to(device)
    
    positive_mask = torch.zeros_like(sim_matrix).to(device)
    for i, label in enumerate(labels.cpu().numpy()):
        if label in proto_labels:
            proto_idx = proto_labels.index(label)
            positive_mask[i, proto_idx] = 1.0
    
    logits_mask = torch.ones_like(sim_matrix).to(device)
    exp_logits = torch.exp(sim_matrix) * logits_mask
    log_prob = sim_matrix - torch.log(exp_logits.sum(1, keepdim=True) + 1e-8)
    mean_log_prob_pos = (positive_mask * log_prob).sum(1) / (positive_mask.sum(1) + 1e-8)
    loss = -mean_log_prob_pos.mean()
    
    if torch.isnan(loss).any() or torch.isinf(loss).any():
        print("Warning: prototype_contrastive_loss contains NaN or Inf, returning 0.0")
        return torch.tensor(0.0, requires_grad=True).to(device)
    
    return loss

def compute_samples_per_cls(data_loader, unique_labels, device='cuda'):
    label_counts = torch.zeros(len(unique_labels)).to(device)
    for _, labels in data_loader:
        labels = labels.to(device)
        label_counts.index_add_(0, labels, torch.ones_like(labels, dtype=torch.float))
    return label_counts.cpu().numpy()

def focal_loss(labels, logits, alpha, gamma):
    BCLoss = F.binary_cross_entropy_with_logits(input=logits, target=labels, reduction="none")
    if gamma == 0.0:
        modulator = 1.0
    else:
        modulator = torch.exp(-gamma * labels * logits - gamma * torch.log(1 + torch.exp(-1.0 * logits)))
    loss = modulator * BCLoss
    weighted_loss = alpha * loss
    focal_loss = torch.sum(weighted_loss)
    focal_loss /= torch.sum(labels)
    return focal_loss

def CB_loss(labels, logits, samples_per_cls, no_of_classes, loss_type, beta, gamma, device='cuda'):
    weights = np.ones(no_of_classes, dtype=np.float32)
    non_zero_mask = samples_per_cls > 0
    effective_num = np.ones(no_of_classes, dtype=np.float32)
    effective_num[non_zero_mask] = 1.0 - np.power(beta, samples_per_cls[non_zero_mask])
    effective_num = np.where(effective_num == 0, 1e-6, effective_num)
    
    weights[non_zero_mask] = (1.0 - beta) / effective_num[non_zero_mask]
    weights_sum = np.sum(weights[non_zero_mask]) if np.sum(non_zero_mask) > 0 else 1.0
    weights = weights / (weights_sum + 1e-6) * no_of_classes
    
    weights = torch.tensor(weights, dtype=torch.float32, device=device)
    labels_one_hot = F.one_hot(labels, no_of_classes).float().to(device)
    
    weights = weights.unsqueeze(0).repeat(labels_one_hot.shape[0], 1) * labels_one_hot
    weights = weights.sum(1)
    weights = weights.unsqueeze(1).repeat(1, no_of_classes)
    
    if loss_type == "focal":
        cb_loss = focal_loss(labels_one_hot, logits, weights, gamma)
    elif loss_type == "sigmoid":
        cb_loss = F.binary_cross_entropy_with_logits(input=logits, target=labels_one_hot, weight=weights)
    elif loss_type == "softmax":
        pred = logits.softmax(dim=1)
        cb_loss = F.binary_cross_entropy(input=pred, target=labels_one_hot, weight=weights)
    
    if torch.isnan(cb_loss).any():
        print("Warning: cb_loss contains NaN, returning 0.0")
        return torch.tensor(0.0, requires_grad=True).to(device)
    
    return cb_loss

def cb_loss(labels, logits, samples_per_cls, unique_labels, loss_type="focal", beta=0.9999, gamma=2.0, device='cuda'):
    no_of_classes = len(unique_labels)
    if logits.size(1) != no_of_classes:
        linear_layer = nn.Linear(logits.size(1), no_of_classes).to(device)
        logits = linear_layer(logits)  
    
    cb_loss_value = CB_loss(labels, logits, samples_per_cls, no_of_classes, loss_type, beta, gamma)
    
    return cb_loss_value