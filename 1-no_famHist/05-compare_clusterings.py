import os
import shutil
import numpy as np
import pandas as pd
import random
import matplotlib.pyplot as plt
import seaborn as sns

#pd.set_option('display.max_rows', 1500)
#pd.set_option('display.max_columns', 1500)

from statistics import median
from sklearn.model_selection import train_test_split
from sklearn.metrics.cluster import adjusted_mutual_info_score
from sklearn.cluster import spectral_clustering
#import netneurotools
from netneurotools import cluster
from snf import compute

#==================================================================================================================

def compute_confusion_matrix(res, file, argmax_stab, argmax_nCl, ld1, ld2, s1, s2):
    patno_cluster_a = pd.read_csv("files/res/02_clusters_runid="+str(argmax_stab)+".csv", index_col=0)
    patno_cluster_b = pd.read_csv("files/res/02_clusters_runid="+str(argmax_nCl)+".csv", index_col=0)
    """
    patno_cluster_b = patno_cluster_b.replace(1,30)
    patno_cluster_b = patno_cluster_b.replace(3,10)
    patno_cluster_b = patno_cluster_b.replace(10,1)
    patno_cluster_b = patno_cluster_b.replace(30,3)
    #"""
    a = res['nb_clusters'].loc[argmax_stab]
    b = res['nb_clusters'].loc[argmax_nCl]
    #print(a,b)
    confusion_matrix = np.matrix(np.zeros((a,b)))
    for cla in range(a):
        for clb in range(b):
            confusion_matrix[cla,clb] = len(set(patno_cluster_a[patno_cluster_a['group'] == cla+1].index) & set(patno_cluster_b[patno_cluster_b['group'] == clb+1].index))
    plt.figure()
    sns.heatmap(confusion_matrix.T, square = True, annot=True, xticklabels = ["cluster "+str(ii) for ii in ['A', 'B', 'C','D', 'E'][:a]], yticklabels = ["cluster "+str(ii) for ii in range(1,b+1)], fmt = '.0f' )
    plt.title('dim=('+ str(nb_patients)+','+str(nb_features)+'), latent_dims = '+ str(ld1)+' & '+str(ld2)
              +', stab = '+ str(s1)+' & '+str(s2))
    plt.margins(x=0.1, y=0.1)
    plt.savefig(file, bbox_inches='tight', dpi=300)
    return confusion_matrix

#==================================================================================================================
def compute_percentage_confusion_matrix(confusion_matrix, file):
    percentage_confusion_matrix = 100*confusion_matrix/np.sum(confusion_matrix, axis = 0)
    a,b = np.shape(confusion_matrix)
    plt.figure()
    sns.heatmap(percentage_confusion_matrix.T, square = True, annot=True, xticklabels = ["cluster "+str(ii) for ii in ['A', 'B', 'C','D', 'E'][:a]], yticklabels = ["cluster "+str(ii) for ii in range(1,b+1)], fmt = '.0f' )
    plt.title('Percentage of cluster intersection (patients,features) = ('+ str(nb_patients)+','+str(nb_features)+')')
    plt.margins(x=0.1, y=0.1)
    plt.savefig(file, bbox_inches='tight', dpi=300)
    return percentage_confusion_matrix

#==================================================================================================================

bl_6m = pd.read_csv('/shared-data/research/genomics/projects/ppmi_analysis/clinical_longitudinal_VAE/bl_6m/02_patno_cluster_runid=183.csv')

nb_patients = int(np.load("files/nb_patients.npy"))
nb_features = int(np.load("files/nb_features.npy"))

"""
index_list =[100, 136, 142, 166, 238, 244, 418, 442, 484, 490, 508, 520, 532, 562, 568, 
                640, 730, 736, 766, 778, 802, 808, 820, 826, 844, 916, 928, 964, 970, 1066, 
                1090, 1120, 1138, 1144, 1180, 1198, 1216, 1240, 1288, 1384, 1414, 1420, 1426, 
                1456, 1468, 1498, 1504, 1564, 1606, 1708, 1780, 1786, 1804, 1822, 1828, 1852, 
                1864, 2032, 2062, 2098, 2128, 2170, 2236, 2242]
                """


index_list =[2032, 1066, 730, 1138, 820, 2098, 826, 1420, 136, 142, 1180, 2128, 1216, 1852, 2170, 964]
res = pd.read_csv("summary_dim="+str(nb_patients)+"_"+str(nb_features)+".csv", index_col=0)
#res2 = res[(res['stability']>=.95) & (res['kl_div_loss_weight']>.1) & (res['latent_dim']>3)]# ]# & (res['latent_dim']==10)]
res2 = res[res.index.isin(index_list)]
res2 = res2[(res2['stability']>=.95) & (res2['kl_div_loss_weight']>.1) & (res['latent_dim']==6)]
#res2 = res2[(res2.nb_clusters<4) & (res2.latent_dim>7)]

if os.path.exists('figs/selected_losses')==False:
    os.mkdir('figs/selected_losses')
else:  
    shutil.rmtree('figs/selected_losses')
    os.mkdir('figs/selected_losses')

index_list = res2.index.tolist()

import shutil
for ind in index_list:
    shutil.copy("figs/loss/02_train_test_loss_id="+str(ind)+".png", "figs/selected_losses/02_train_test_loss_id="+str(ind)+".png")


"""
src_dir = "figs/loss"
dst_dir = "figs/conserved_loss"
if os.path.exists(dst_dir):
    shutil.rmtree(dst_dir)
os.mkdir(dst_dir)
all_ids = res2.index.tolist()
for idf in all_ids:
    substring = 'id='+str(idf)
    for pngfile in glob.iglob(os.path.join(src_dir, "*.png")):
        if substring in pngfile:
            shutil.copy(pngfile, dst_dir)
"""

#res2.to_csv('valid_summary.csv')
res2 = res2.sort_values("stability", ascending = False)
#df_all_argmax_stab = res2[res['stability']>=1-1e-6]
#argmax_stab = df_all_argmax_stab['nb_clusters'].idxmax() #258 , 26, 338, 17, 50
argmax_stab = res2['stability'].idxmax()
argmax_nCl = res2['nb_clusters'].idxmax()
print("---------")
print(res.loc[[argmax_stab, argmax_nCl]])
print("---------")

#
#"""
file = 'figs/05-cluster_intersection_'+str(argmax_stab)+'_'+str(argmax_nCl)+'.png'
confusion_matrix = compute_confusion_matrix(res, file, argmax_stab, argmax_nCl, 
                                        int(res.loc[argmax_stab].latent_dim), int(res.loc[argmax_nCl].latent_dim), 
                                        round(res.loc[argmax_stab].stability,2), round(res.loc[argmax_nCl].stability,2))
#file = 'figs/05-percentage_cluster_intersection.png'
#_ = compute_percentage_confusion_matrix(confusion_matrix, file)
#"""
res_5c = res2[res2['nb_clusters'].isin([2,5])]

import os, shutil
if os.path.exists('figs/confusion')==True:
    shutil.rmtree('figs/confusion')
os.mkdir('figs/confusion')

count = 0
indices_5c = res_5c.index.tolist()
for ii in indices_5c:
    for jj in indices_5c:
        if ii<jj and res_5c['nb_clusters'][ii] != res_5c['nb_clusters'][jj]:
            file = 'figs/confusion/05-cluster_intersection_'+str(ii)+'_'+str(jj)+'.png'
            confusion_matrix = compute_confusion_matrix(res, file, ii, jj, 
                                        int(res_5c.loc[ii].latent_dim), int(res_5c.loc[jj].latent_dim), 
                                        round(res_5c.loc[ii].stability,2), round(res_5c.loc[jj].stability,2))
            count = count+1
print("count = ", count)
print(res_5c)

"""
fig, ax = plt.subplots()
ax.barh(np.arange(1,6), confusion_matrix[0,:].tolist()[0], height=0.3, color='blue', label='Cluster A')
ax.barh(np.arange(1,6), (-confusion_matrix[1,:]).tolist()[0], height=0.3, color='magenta', label='Cluster B')
plt.xlim(-145,105)
plt.yticks(range(1,6), ['Cluster '+str(ii) for ii in range(1,6)])
# set title and show the diagram
plt.title('Cluster structure')
#plt.xlim([-90, 105])
ax.bar_label(ax.containers[0],fontsize= 12)
ax.bar_label(ax.containers[1],fontsize= 12)
plt.legend()
plt.savefig('figs/05-2_vs_5.png', dpi = 300)


For stacked bar plots
plt.bar(x, y1, color='r')
plt.bar(x, y2, bottom=y1, color='b')
"""
"""
patno_cluster_a = pd.read_csv("files/res/02_patno_cluster_runid=2127.csv", index_col=0)
patno_cluster_b = pd.read_csv("files/res/02_patno_cluster_runid=255.csv", index_col=0)
#patno_cluster_b = bl_6m.set_index('PATNO')
a = 5
b = 5
confusion_matrix = np.matrix(np.zeros((a,b)))
for cla in range(a):
    for clb in range(b):
        confusion_matrix[cla,clb] = len(set(patno_cluster_a[patno_cluster_a['group'] == cla+1].index) & set(patno_cluster_b[patno_cluster_b['group'] == clb+1].index))
plt.figure()
sns.heatmap(confusion_matrix.T, square = True, annot=True, xticklabels = ["cluster "+str(ii) for ii in ['A', 'B', 'C','D', 'E'][:a]], yticklabels = ["cluster "+str(ii) for ii in range(1,b+1)], fmt = '.0f' )
plt.title('longitudinal vs. BL')
plt.margins(x=0.1, y=0.1)
file = 'figs/05-bl_6m_compare.png'
plt.savefig(file, bbox_inches='tight', dpi=300)
"""