# Torch and torchvision imports
import sys
import logging
from datetime import datetime
import torch
import torch.utils.data
from torch.utils.data import DataLoader
from torch import nn, optim
from torch.nn import functional as F

# Basic data science imports
import os
import json
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# Sklearn scipy, statsmodels
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics.cluster import adjusted_mutual_info_score
from sklearn.metrics import r2_score

from sklearn.cluster import spectral_clustering
from statistics import median
from tqdm import tqdm
import pickle

from netneurotools import cluster
import netneurotools 

from snf import compute

from py_pcha import PCHA

from sklearn.manifold import TSNE

#Set Seeds
np.random.seed(1)
torch.manual_seed(1)
random_seed = 1
chosen_metric = 'cosine'
#---------------------------------------------------------------------
# VAE and Clustering Functions
# Variational Autoencoder Architecture
class VAE(nn.Module):
    def __init__(self, input_dim, hidden_dim, latent_dim):
        super(VAE, self).__init__()
        
        self.fc1 = nn.Linear(input_dim, hidden_dim) 
        self.fc21 = nn.Linear(hidden_dim, latent_dim)
        self.fc22 = nn.Linear(hidden_dim, latent_dim)
        self.fc3 = nn.Linear(latent_dim, hidden_dim)
        self.fc4 = nn.Linear(hidden_dim, input_dim)

    def encode(self, x):
        h1 = nn.functional.relu(self.fc1(x)) 
        mu = self.fc21(h1)
        logvar = self.fc22(h1)
        return mu, logvar
    
    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, z):
        h2 = nn.functional.relu(self.fc3(z))
        return torch.sigmoid(self.fc4(h2))

    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        recon_x = self.decode(z)
        return recon_x, mu, logvar
#---------------------------------------------------------------------
"""class LogCoshLoss(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, y_t, y_prime_t):
        ey_t = y_t - y_prime_t
        return torch.mean(torch.log(torch.cosh(ey_t + 1e-12)))"""
#---------------------------------------------------------------------
# Loss function
def loss_function(recon_x, x, mu, logvar):
    reconst_loss = nn.functional.mse_loss(recon_x, x, reduction='sum') 
    KL_div_loss = - 0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp()) # D_KL( q_phi(z | x) || p_theta(z) )

    return reconst_loss, KL_div_loss

#---------------------------------------------------------------------
# Define training iteration function
def train(model, device, optimizer, kl_div_loss_weight = 0.1):
    """
    kl_div_loss_weight : float (optional)
        controls the balance between reconst_loss and KL div (higher means more weight
        on the KL_div part of the loss). This is Beta in the MVAE paper
    """
    model.train() 
    
    train_loss = 0
    running_reconst_loss = 0.0
    running_KL_div_loss = 0.0

    for batch_idx, data in enumerate(train_loader):
        data = data.to(device)
        optimizer.zero_grad()
        recon_batch, mu, logvar = model(data)
        reconst_loss, KL_div_loss = loss_function(recon_batch, data, mu, logvar)
        loss = reconst_loss + kl_div_loss_weight * KL_div_loss
        loss.backward()

        # Update running loss totals 
        train_loss += loss.item()
        running_reconst_loss += reconst_loss.item()
        running_KL_div_loss += KL_div_loss.item()

        optimizer.step()

    total_loss = train_loss / len(train_loader.dataset) # divide by number of training samples
    reconst_loss = running_reconst_loss / len(train_loader.dataset)
    KL_div_loss =  running_KL_div_loss / len(train_loader.dataset)

    return total_loss, reconst_loss, KL_div_loss
#---------------------------------------------------------------------

# Define test iteration function
def test(model, device, kl_div_loss_weight = 0.1):
    model.eval() 
    test_loss = 0
    with torch.no_grad():
        for batch_idx, data in enumerate(test_loader):
            data = data.to(device)
            recon_batch, mu, logvar = model(data)
            reconst_loss, KL_div_loss = loss_function(recon_batch, data, mu, logvar)
            loss = reconst_loss + kl_div_loss_weight * KL_div_loss

            test_loss += loss.item()
    
    test_loss /= len(test_loader.dataset)

    return test_loss
#---------------------------------------------------------------------

#Spectral Clustering
def cluster_labels(embed):
    affinities = compute.make_affinity(embed, metric=chosen_metric)
    #print("affinities = ", affinities)
    first, second = compute.get_n_clusters(affinities)
    fused_labels = spectral_clustering(affinities, n_clusters=first)#this is not deteministic
    #print('second : ', second)
    return fused_labels,first
#---------------------------------------------------------------------

#Consensus Clustering
def consenus(embed, n):
    
    ci=[]
    
    for i in range(n):
        fused_labels,first=cluster_labels(embed)#could be optimized to compute affinities and n_clusters only once
        ci.append(list(fused_labels))
        
    consensus = cluster.find_consensus(np.column_stack(ci), seed=random_seed)
    a,=np.unique(consensus).shape
    
    return consensus,a
#---------------------------------------------------------------------

# Compute Clustering Stability
def ami_cluster_stability(data, true_labels, k, split= 0.20):
    X_sample, X_rest, y_sample, y_rest = train_test_split(data, true_labels, test_size=split)
    affinities = compute.make_affinity(X_sample, metric=chosen_metric)
    y_cluster = spectral_clustering(affinities, n_clusters=k)
    return adjusted_mutual_info_score(y_cluster, y_sample)
#---------------------------------------------------------------------

def calculate_metrics(data, n_clusters, labels, iterations=20):
    cluster_stability = []

    for i in range(iterations):
        cluster_stability.append(ami_cluster_stability(data, labels, n_clusters))
    
    return cluster_stability
#---------------------------------------------------------------------

#Get Data
X1 = pd.read_csv("files/median_imputed_df_for_VAE.csv", index_col=0)

# Min-Max Scale the data
scaler = MinMaxScaler()
X_scaled1 = scaler.fit_transform(X1)

# Get only the clinical-scale features (omitting those marked as Covariate, if applicable)
df1 = pd.DataFrame(X_scaled1, columns=X1.columns, index=X1.index)
scales_columns = df1.columns.tolist()#[ df1.columns.str.startswith( 'Protein' ) ].tolist()

df = df1[scales_columns]
X_scaled = df.values

n_participant, input_dim = X_scaled.shape
#print("(n_patnos, input_dim) = (", n_participant, input_dim, ")")
#print(X1.columns)

X_train, X_val = train_test_split(X_scaled, train_size = 0.8, shuffle=True)
X_train_tensor, X_val_tensor = torch.Tensor(X_train), torch.Tensor(X_val)
BATCH_SIZE = 64

train_loader = DataLoader(
    X_train_tensor,
    batch_size=BATCH_SIZE,
    shuffle=True
)

test_loader = DataLoader(
    X_val_tensor,
    batch_size=BATCH_SIZE,
    shuffle=False
)
#Initiate VAE
ii = 0
hyperparams = []
for line in open("tune_vae.txt"):
    txt_row = line.split()
    logging.info(str(txt_row))
    hyperparams.append(txt_row[1])
    ii = ii + 1

run_id = int(hyperparams[0])
latent_dim = int(hyperparams[1])
hidden_dim = int(hyperparams[2])
epochs = int(hyperparams[3])
learning_rate = 10**int(hyperparams[4])
kl_div_loss_weight = float(hyperparams[5])

#device = torch.device("cuda" )
device = torch.device("cpu" ) 
model =  VAE(input_dim, hidden_dim, latent_dim).to(device)
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

# Train VAE
train_loss_all_epochs = []
reconst_loss_all_epochs = []
KL_div_loss_all_epochs = []
test_loss_all_epochs = []

for epoch in range(0, epochs):
    total_loss, reconst_loss, KL_div_loss = train(model, device, optimizer, kl_div_loss_weight)
    test_loss = test(model, device, kl_div_loss_weight=kl_div_loss_weight)
    train_loss_all_epochs.append(total_loss)
    reconst_loss_all_epochs.append(reconst_loss)
    KL_div_loss_all_epochs.append(KL_div_loss)
    test_loss_all_epochs.append(test_loss)
# Create embedding for all participants
data1 = torch.Tensor(X_scaled).to(device)
mu, logvar = model.encode(data1)
embed = mu.cpu().detach().numpy()

projection = pd.DataFrame(embed, index=df.index)


#"""
#Visualizing the projection
plt.figure()
sns.set(rc={'figure.figsize':(10,15)})
ax=sns.heatmap(embed, cmap='bwr',center=0.00,)
ax.set(xlabel='Embedding', ylabel='Participant')
ax.set_title('Projection of participants in latent space')
plt.savefig('figs/participants_by_cluster/02_proj_in_latent_space.png')
#"""
#Perform SVD
from numpy.linalg import svd
cols = X1.columns
U, sgv, V = svd(X1[cols].values)
#print('singular values', sgv)
cum_sgv = np.cumsum(sgv)/np.sum(sgv)
rank_80 = next(x for x, val in enumerate(cum_sgv) if val > 0.8)
rank_90 = next(x for x, val in enumerate(cum_sgv) if val > 0.9)
rank_95 = next(x for x, val in enumerate(cum_sgv) if val > 0.95)
nb_sgv = min(X1.shape)
plt.figure()
plt.plot(np.arange(nb_sgv)+1, sgv, color = 'b', linewidth = 1.5)
plt.yscale('log')
plt.xlabel('rank')
plt.ylabel('singular value')
plt.axvline(x = rank_80, color = 'y', label = '80% cumulative threshold', linewidth = 1.5)
plt.axvline(x = rank_90, color = 'r', label = '90% cumulative threshold', linewidth = 1.5)
plt.axvline(x = rank_95, color = 'g', label = '95% cumulative threshold', linewidth = 1.5)
plt.legend()
plt.savefig('figs/02_sgv_clinical.png')
#"""
 
# R^2 on all participants (train and validation)
reconstruction = model.decode(mu)
r2_train_valid = r2_score(data1.cpu().detach().numpy(), reconstruction.cpu().detach().numpy(), multioutput='variance_weighted')
#print("train & validation r2_score : "+str(r2_train_valid))
"""
if r2_train_valid<0:
    print("Exiting script because an R2<0 error")
    sys.exit()
"""

# R^2 on validation set
mu_val, _ = model.encode(X_val_tensor.to(device))
reconst_val = model.decode(mu_val)
r2_validation_variance_weighted = r2_score(X_val_tensor.cpu().detach().numpy(), reconst_val.cpu().detach().numpy(), multioutput='variance_weighted')
print("train+valid R2 : "+str(round(r2_train_valid,3))+" --- valid R2 : "+str(round(r2_validation_variance_weighted,3)))
if r2_validation_variance_weighted<0:
    print("Exiting script because an R2<0 error")
    sys.exit()


fused_labels, clusters = consenus(embed, 30)


cluster_stability = calculate_metrics(embed, clusters, fused_labels)
stability = median(cluster_stability)

print('Number of clusters: ' + str(clusters) + ' ----- Clustering Stability: '+ str(round(stability,2)))
print('---------------------------------------------------------- ')

nb_patients, nb_features = X1.shape

#"""
from math import log10
f = open("summary_dim="+str(nb_patients)+"_"+str(nb_features)+".csv", "a")
f.write(str(run_id)+","+str(latent_dim)+","+str(hidden_dim)+","+str(epochs)+","
        +str(log10(learning_rate))+","+str(kl_div_loss_weight)+","+str(clusters)
        +","+str(stability)+","+str(r2_validation_variance_weighted)+"\n")
f.close()
#"""

# Distribution of clusters
projection['group'] = fused_labels
projection['group'].value_counts()


#""" 
# Plot loss
plt.figure()
sns.set(rc={'figure.figsize':(8,6)})
plt.plot(train_loss_all_epochs,label='Total Train Loss')
plt.plot(reconst_loss_all_epochs, label='Reconstruction Loss')
plt.plot(KL_div_loss_all_epochs, label='KLD Loss')
plt.plot(test_loss_all_epochs, label='Test Loss')
plt.title('Loss During Training')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend() 
plt.savefig('figs/loss/02_train_test_loss_id='+str(run_id)+'.png')
#"""
#Visualization of clusters in 2D using TSNE
#"""
plt.figure()
sns.set(rc={'figure.figsize':(6,6)})

tsne = TSNE(n_components=2,n_jobs=-1)
tsne_results = tsne.fit_transform(embed)

ax=sns.scatterplot(
    x=tsne_results[:,0], y=tsne_results[:,1], hue=projection['group'], palette="deep")

ax.set(xlabel='TSNE 1', ylabel='TSNE 2')
plt.savefig('figs/tsne/02_tsne_id='+str(run_id)+'.png')
#"""

"""
# Look at by-feature R^2
r2_raw = r2_score(X_val_tensor.cpu().detach().numpy(), reconst_val.cpu().detach().numpy(), multioutput='raw_values')
r2_raw = pd.Series(r2_raw, index=df.columns)

with open('files/colnames_for_plots.json', 'r') as openfile:
    colnames_for_plots = json.load(openfile)

plt.figure()
r2_raw.rename(index=colnames_for_plots, inplace=True)
fig, ax = plt.subplots(figsize=(8,14))
r2_raw.sort_values()
r2_raw.sort_values(ascending=False)[0:100].plot.barh()
plt.title(f"Reconstruction R2 Score By Feature (validation set) \n Variance-weighted avg. = {r2_validation_variance_weighted:0.2f}", fontsize=18)
plt.gcf().savefig('figs/02_feature_reconstruction.png', bbox_inches='tight')
#"""
iter1={}
iter1['projection']=projection

#Projections on archetypes + Clusters
df_dbs=iter1['projection']
df_dbs= df_dbs.add_prefix('Projection_')
df_dbs= df_dbs.rename(columns={'Projection_group': 'group'})
#print("Projections and clusters :\n", df_dbs)

#"""
group_counts = df_dbs['group'].value_counts().sort_index(ascending=False).sort_index()#.plot.barh()
#Vertical orientation
fig, ax = plt.subplots()
ax = sns.barplot(x=group_counts.index, y=group_counts.values)
ax.bar_label(ax.containers[0],fontsize= 20)
ax.set_xticklabels([f'Cluster {i}' for i in group_counts.index], fontsize=14)
#ax.set_yticks([])
#ax.set_ylim([0,105])
#ax.set_ylabel("# participants", fontsize=20)
ax.set_title("Number of participants per cluster", fontsize=24)
sns.despine(bottom = True, left = True)
plt.xticks(rotation=45)

fig.savefig('figs/participants_by_cluster/02_cluster_count_plot_id='+str(run_id)+'.png', dpi=300, bbox_inches='tight')
#"""

#"""

plt.figure()
sns.set(rc={'figure.figsize':(6,6)})
fig = plt.figure()
sns.set_palette('tab10')
sns.set_color_codes("muted")
ax=sns.scatterplot(x=embed[:,0], y=embed[:,1], hue=df_dbs['group'], palette='tab10')

plt.legend(bbox_to_anchor=(1.02, 1), loc='upper left', borderaxespad=0, prop={'size': 16})

plt.xlabel('UMAP 0', fontsize=24)
plt.ylabel('UMAP 1', fontsize=24)
plt.title('Clinical VAE Clusters', fontsize=24)
#ax.set_xticks([0,2,4,6,8],fontsize=18)
#ax.set_yticks([0,2,4,6,8],fontsize=18)
fig.savefig('figs/umap/02_UMAP_clusters_id='+str(run_id)+'.png')#, dpi=1000, bbox_inches='tight')


#"""
df_cluster = pd.DataFrame(embed, index=df.index)
df_cluster['group'] = projection['group']
df_cluster.to_csv("files/res/02_clusters_runid="+str(run_id)+".csv")