import quilt3

# Package names should follow this standard: {Company}/{Dataset-Indication}
package_name = "NMRA/AnswerALS-ALS"

# Refers to the S3 bucket that will store the package
registry = "s3://trestle-curated"

# Pull the latest version of the package from remote storage (S3)
p = quilt3.Package.browse(package_name, registry=registry)

# Add a file to the package or update an existing one. 
# The p.set() function takes two args:
# 1) Where the file exists within the package
# 2) Where the file exists outside of the package
#p.set("1-no_famHist/initial-data/aals_phenodata_n1041_v4.csv", "aals_phenodata_n1041_v4.csv")
#p.delete("1-no_famHist/initial-data/aals_phenodata_n1041_v4.csv")

p.set("clinical/als_only/initial-data/aals_phenodata_n1041_v4.csv", "../initial_data/aals_phenodata_n1041_v4.csv")
p.set("clinical/als_only/initial-data/aals_dx_n1041.csv", "../initial_data/aals_dx_n1041.csv")

p.set("clinical/als_only/intermediate-data/median_imputed_df_for_VAE.csv", "files/median_imputed_df_for_VAE.csv")

"""
p.set("1-no_famHist/intermediate-data/median_imputed_df_for_VAE.csv", "files/median_imputed_df_for_VAE.csv")
p.set("1-no_famHist/results/tune_vae.txt", "tune_vae.txt")
"""

p.set("clinical/als_only/1-no_famHist/ReadMe.txt", "ReadMe.txt")

p.set("clinical/als_only/1-no_famHist/results/clusters_runid=2098.csv", "files/res/02_clusters_runid=2098.csv")
#p.set("clinical/als_only/1-no_famHist/results/train_test_loss_runid=2098.csv", "figs/loss/02_train_test_loss_id=2098.png")

p.set("clinical/als_only/1-no_famHist/results/clusters_runid=820.csv", "files/res/02_clusters_runid=820.csv")
#p.set("clinical/als_only/1-no_famHist/results/train_test_loss_runid=820.csv", "figs/loss/02_train_test_loss_id=820.png")

p.set("clinical/als_only/1-no_famHist/results/hierarchical_structure_(MILD).png", "figs/confusion/05-cluster_intersection_820_2098.png")
# Let people know whom to contact for questions about the files added/changed
p.set_meta({'author': 'amina.benaceur@neumoratx.com', 
            'code': 'https://github.com/nmra-abenaceur/als_only_clinical_vae/1-no_famHist'})

# How is this package changing? Complete the sentence: 'This commit will...'
#message = "1st commit of ALS clinical BL clustering with item-level features"
message = "1st upload ALS only patients"

# Save your changes remotely, similar to `git push origin HEAD`
p.push(package_name, registry=registry, message=message)


