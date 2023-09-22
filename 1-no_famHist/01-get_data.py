import os
import numpy as np
import pandas as pd
pd.set_option('display.max_rows', 1500)
pd.set_option('display.max_columns', 1500)

from statistics import median

#--------------------------------------------------------
def impute_medians(df):
      df_fillna = df.copy(deep=True)
      for col in df.columns:
            #print('col :', col)
            med = median(df[col][df[col].notnull()])
            df_fillna[col] = df_fillna[col].fillna(med)
      return df_fillna
#--------------------------------------------------------
if os.path.exists('figs')==False:
    os.mkdir('figs')
if os.path.exists('files')==False:
    os.mkdir('files')
if os.path.exists('files/res')==False:
    os.mkdir('files/res')
if os.path.exists('figs/loss')==False:
    os.mkdir('figs/loss')
if os.path.exists('figs/tsne')==False:
    os.mkdir('figs/tsne')
if os.path.exists('figs/umap')==False:
    os.mkdir('figs/umap')
if os.path.exists('figs/participants_by_cluster')==False:
    os.mkdir('figs/participants_by_cluster')
#--------------------------------------------------------

raw_data = pd.read_csv("../initial_data/aals_phenodata_n1041_v4.csv")
disease_type_data = pd.read_csv("../initial_data/aals_dx_n1041.csv")
als_data = disease_type_data[disease_type_data['Cohort']=='ALS']

df = raw_data.copy()
print(df.shape)

als_case_ids = list(set(als_data['Participant_ID'].values).intersection(set(df['Participant_ID'].values)))
df = df[df['Participant_ID'].isin(als_case_ids)]

df = df[df['Case_Control']=='Case']
df = df.drop(['Unnamed: 0', 'Case_Control', 'Ethnicity', 'Black', 'Asian', 'AmerIndian'], axis=1).set_index('Participant_ID')
print(df.shape)
#################################
### checking missingness in curated data cut 

missingness_cols = ((df.isnull().sum()/len(df))*100).sort_values(ascending=False)
#print(missingness_cols)

missingness_rows = ((df.T.isnull().sum()/len(df.T))*100).sort_values(ascending=False)
#print(missingness_rows)
#print(df.shape)

patients_lack_info = df.copy()
patients_lack_info['missingness'] = missingness_rows
patients_lack_info = patients_lack_info[patients_lack_info['missingness']>=50]
patients_lack_info = patients_lack_info.sort_values(by = 'missingness', ascending=False)

df['NIV_Use'] = df['NIV_Use'].fillna('300').astype('int')
df['PAV_Use'] = df['PAV_Use'].fillna('300').astype('int')
df['pav_niv'] = None
#print('a')
#df['pav_niv'] = df['NIV_Use']
for idx in df.index:
    if df['PAV_Use'].loc[idx] == 1:
        df['pav_niv'].loc[idx] =  2
    else:
        #print('a')
        df['pav_niv'].loc[idx] = df['NIV_Use'].loc[idx]
        #print('b')

df[['NIV_Use', 'PAV_Use', 'pav_niv']].to_csv('check.csv')

df['NIV_Use'] = df['NIV_Use'].replace(300, np.nan)
df['PAV_Use'] = df['PAV_Use'].replace(300, np.nan)
#df[['PAV_Use', 'NIV_Use', 'pav_niv']].to_csv('files/check.csv')
df = df.drop(['NIV_Use', 'PAV_Use'], axis = 1)

df['Sex'] = df['Sex'].replace('Male', 0)
df['Sex'] = df['Sex'].replace('Female', 1)

missingness_cols = ((df.isnull().sum()/len(df))*100).sort_values(ascending=False)

missingness_rows = ((df.T.isnull().sum()/len(df.T))*100).sort_values(ascending=False)

high_miss_features = list(missingness_cols[missingness_cols>=51].index)+['Ashworth_Slope', 'ALSFRSR_Slope'] # 51% TO KEEP SVC(important), THE 2 SLOPES ARE OMITTED BECAUSE WE DO NOT WANT TO CONSIDER PROGRESSION
df = df.drop(high_miss_features, axis=1)
df.to_csv('files/before_median_imputation.csv')

missingness_cols = ((df.isnull().sum()/len(df))*100).sort_values(ascending=False)
print('--------------------------')
print(df.shape)
print(missingness_cols)



median_imputed_df = impute_medians(df)
# Add jitter to columns with only 4 or fewer distinct values

#To increase robustness, we considered only a subset of clinical variables, which is why the 2 lines of code below are commented
#continuous_features = ['age_enrollment', 'VLTANIM', 'VLTVEG', 'VLTFRUIT', 'hvlt_retention', 'DVSD_SDM', 'DVT_SDM', 'DVS_JLO_MSSAE', 'duration']
#integer_features = list(set(median_imputed_df.columns).difference(set(continuous_features)))

#"""
mu, sigma = 0, 0.1

for col in median_imputed_df.columns:
    #print(f"Jittering {col}")
    s = np.random.normal(mu, sigma, median_imputed_df[col].shape[0])
    modified_column = median_imputed_df[col] + s
    median_imputed_df[col] = modified_column
#"""

median_imputed_df.to_csv('files/median_imputed_df_for_VAE.csv')

np.save("files/nb_patients.npy", median_imputed_df.shape[0])
np.save("files/nb_features.npy", median_imputed_df.shape[1])

columns0 = median_imputed_df.columns.to_list()
to_write = "{\n"
for ii, col in enumerate(median_imputed_df.columns.tolist()):
    to_write = to_write + '"' + col + '"' + ": " + '"' + columns0[ii]+ '"' + ",\n"
to_write = to_write[0:-2]
to_write = to_write + "\n}"
f = open("files/colnames_for_plots.json", "w")
f.write(to_write)
f.close()