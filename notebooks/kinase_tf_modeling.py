#!/usr/bin/env python
# coding: utf-8

# # Effect of Phosphorylation on Transcription Factors in *Mycobacterium tuberculosis*

# In[17]:


# Core Library Imports
import os
import json

# External Library Imports
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import optuna
import pandas as pd
import seaborn as sb
import sklearn.model_selection

# Local Imports
import mycokinaseseq.kinase_tf_model
import mycokinaseseq.utils
import mycokinaseseq.list_helper as lh


# In[18]:


# Global variables to determine some features of running this notebook
OPT_HYPERPARAMETERS = True


# ## Create Target Dictionaries
# The first step is to create dictionaries which represent which TFs, and Kinases target which genes

# In[19]:


# Read in the Transcription Factor Overexpression Data
# noinspection PyTypeChecker
tfoe_df = pd.read_excel(os.path.join("..","data", "tfoe.searchable_130115.xlsx"), sheet_name="TFOE.data",
                        header=0, usecols="A:HB", skiprows=9, index_col=0,
                        nrows=(4035-9))
# Rename the index column to be "gene"
tfoe_df.rename_axis('gene', inplace=True)
# Find the transcription factor targets
tf_targets_dict = {}
for tf in  tfoe_df.columns:
    tf_targets_dict[tf] = list(tfoe_df[np.abs(tfoe_df[tf] > 1)][tf].index)


# In[20]:


# Create a dictionary for converting Pkn.. to Rv..
kinase_locus_dict = {
    "PknA": "Rv0015c",
    "PknB": "Rv0014c",
    "PknD": "Rv0931c",
    "PknE": "Rv1743",
    "PknF": "Rv1746",
    "PknG": "Rv0410c",
    "PknH": "Rv1266c",
    "PknI": "Rv2914c",
    "PknJ": "Rv2088",
    "PknK": "Rv3080c",
    "PknL": "Rv2176"
}

# Create a dictionary for converting Rv.. to Pkn..
locus_kinase_dict = {v:k for k,v in kinase_locus_dict.items()}

# Create target information from the overexpression data
# Read in the Kinase DF containing overexpression data
in_file = os.path.join("..","data","Total_pTMT_Ascore19_p0.005_OE.xlsx")
kinase_dict_oe = {}
for kinase in ["B","D","E","F","G","H","I","J","K","L"]:
    df = pd.read_excel(in_file, sheet_name=f"Pkn{kinase}_OE", header=0)
    df.rename(columns={"Rv..":"Gene"}, inplace=True)
    kinase_dict_oe[f"Pkn{kinase}"] = list(df["Gene"])

# Repeat but with knockout/knockdown information
in_file = os.path.join("..", "data", "Total_pTMT_Ascore19_p0.005_KO.xlsx")
kinase_dict_ko = {}
# Since PknB is a KD instead of KO, must be done seperately
df = pd.read_excel(in_file, sheet_name="PknB_KD", header=0)
df.rename({"Rv..":"Gene"}, inplace=True, axis=1)
kinase_dict_ko["PknB"] = list(df["Gene"])
# Now, for the remaining kinases
for kinase in ["D","E","F","G","H","I","J","K","L"]:
    df = pd.read_excel(in_file, sheet_name=f"Pkn{kinase}_KO", header=0)
    df.rename(columns = {"Rv..":"Gene"}, inplace=True)
    kinase_dict_ko[f"Pkn{kinase}"] = list(df["Gene"])

# Combine the KO and OE dictionaries
kinase_targets_dict = {}
for kinase in kinase_dict_oe.keys():
    targets = lh.get_unique_list(kinase_dict_oe[kinase]+kinase_dict_ko[kinase])
    kinase_targets_dict[kinase] = targets


# ## Store Target Dictionaries
# Now that the target dictionaries have been created, they can be stored as json files

# In[21]:


# Save the TF dict
with open(os.path.join("..", "data", "dictionaries", "tf_targets_dict.json"), "w") as f:
    json.dump(tf_targets_dict, f)
# Save the kinase dict
with open(os.path.join("..","data","dictionaries","kinase_targets_dict.json"), "w") as f:
    json.dump(kinase_targets_dict, f)
# Save the kinase locus tag dictionaries as well
with open(os.path.join("..", "data", "dictionaries", "kinase_locus_dict.json"), "w") as f:
    json.dump(kinase_locus_dict, f)
with open(os.path.join("..","data","dictionaries", "locus_kinase_dict.json"), "w") as f:
    json.dump(locus_kinase_dict, f)


# ## Data Preprocessing
# Now that the target information has been gathered, the compendia, and differential kinase expression RNA seq data can be processed

# In[22]:


# Read in the RNA seq compendia for fitting the regression model
rna_seq_compendia = pd.read_csv(os.path.join("..","data","log_tpm.csv"), header=0, index_col=0).transpose()
# Read in the raw counts for the RNA seq compendia
rna_seq_raw = pd.read_csv(os.path.join("..","data","counts.csv"), header=0, index_col=0).transpose().loc[rna_seq_compendia.index, rna_seq_compendia.columns]
# Read in the sample information
sample_information = pd.read_csv(os.path.join("..","data","sample_table.csv"), header=0, index_col=0)
# Read in the RNA seq data from the OD and KD experiments
kinase_rna_seq = pd.read_excel(os.path.join("..","data", "STPK_KO_OE_KD_RNAseqData040221_Rformat.xlsx"),
                             sheet_name="Sheet1", header=0, index_col=0)


# First, the RNA seq compendia needs to have genes where there are more than 10% of samples with low reads, and samples where there are more than 25% low reads. The bound for "low reads" will be determined by examining the histogram of the count data, and finding where the low peak is.

# In[23]:


ul = 20
flattened_rna_seq_raw = rna_seq_raw.to_numpy().flatten()
filtered_rna_seq_raw = flattened_rna_seq_raw[flattened_rna_seq_raw < ul]

fig = plt.figure()
ax = plt.axes()
sb.histplot(filtered_rna_seq_raw, ax=ax, kde=True, stat="density", binwidth=1, discrete=True)
ax.set_xlim((0,ul))
ax.set_xlabel("Counts")
ax.set_title("Density of Counts")


# In[24]:


# Based on the above histogram, a lower bound of 3 reads was chosen.
gene_lower_bound_counts = 3
gene_small_proportion = 0.1
sample_lower_bound_counts = 3
sample_small_proportion = 0.25

# Start by removing the samples where more than sample_small_proportion are less than sample_lower_bound_counts
small_count_samples = (rna_seq_raw<=gene_lower_bound_counts).mean(axis=1)
small_count_samples = small_count_samples[small_count_samples>sample_small_proportion].index
updated_rna_seq_raw = rna_seq_raw.drop(small_count_samples, axis=0)

# Next, remove genes where more than gene_small_proportion are less than gene_lower_bound_counts
small_count_genes = (rna_seq_raw<=gene_lower_bound_counts).mean(axis=0)
small_count_genes = small_count_genes[small_count_genes>gene_small_proportion].index
updated_rna_seq_raw = updated_rna_seq_raw.drop(small_count_genes, axis=1)

# Update the gene compendia to only include those genes and samples with sufficiently high reads
filtered_rna_compendia = rna_seq_compendia.loc[updated_rna_seq_raw.index, updated_rna_seq_raw.columns]


# The reference conditions are also still included in the rna_compendia, and those will need to be removed

# In[25]:


# Create dictionaries to hold information about which samples are in which project, and what the reference conditions for that project are
project_to_samples = {}
samples_to_project = {}
project_to_reference = {}
project_to_reference_samples ={}
reference_samples = []
# Go through each project and fill the dictionaries with the information
for project, df in sample_information.groupby("project"):
    # Get the name of the reference condition
    ref_condition = df["reference_condition"].values[0]
    # Check if all reference conditions are the same (they should be)
    if not (df["reference_condition"] == ref_condition).all():
        print(f"Project {project} has more than 1 reference condition")
        break
    # Add information to the dictionaries
    project_to_reference[project]=ref_condition
    project_to_samples[project] = list(df.index)
    for sample in df.index:
        samples_to_project[sample]=project
    # Find all the reference condition samples
    references = df[df["condition"]==ref_condition]
    project_to_reference_samples[project]=list(references.index)
    reference_samples+=list(references.index)


# In[26]:


# Now, remove the reference samples from the filtered compendia
samples_to_remove = lh.find_intersect(list(filtered_rna_compendia.index), reference_samples)
filtered_rna_compendia = filtered_rna_compendia.drop(samples_to_remove, axis=0)


# The kinase experiment data are in the form of RPKM, and must be converted to TPM, and then log2(fold-change) form

# In[27]:


# Convert the RPKM to TPM using formula from Zhao et al., 2020 RNA
kinase_rna_seq = mycokinaseseq.utils.rpkm_to_tpm(kinase_rna_seq)
# For OE, the control should be the pEXCF (First 3 rows), for the KO/KD the control should be WT_1, WT_2, WT_3,
# WT1_Stationary, WT2_Stationary, and WT3_Stationary
# Create dictionaries to keep this information
reference_conditions = {
    "OE":["pEXCF_Stat_1","pEXCF_Stat_2","pEXCF_Stat_3"],
    "KO":["WT_1","WT_2","WT_3","WT1_Stationary","WT2_Stationary","WT3_Stationary"]
}
# Create a dictionary with series to hold the average expression in the reference conditions
reference_condition_expression = {
    "OE": kinase_rna_seq.loc[reference_conditions["OE"]].mean(),
    "KO": kinase_rna_seq.loc[reference_conditions["KO"]].mean()
}
# Create a dictionary to map from expression change ("OE", "KO") to the samples
differential_kinase_conditions = {
    "OE":["PknBOE_1", "PknBOE_2", "PknBOE_3",
          "PknD_OE_1","PknD_OE_2","PknD_OE_3",
          "PknEOE_1","PknEOE_2","PknEOE_3",
          "PknFOE1_Stationary", "PknFOE2_Stationary", "PknFOE3_Stationary",
          "PknGOE1_Stationary", "PknGOE2_Stationary", "PknGOE3_Stationary",
          "PknH_OE_1", "PknH_OE_2", "PknH_OE_3",
          "PknIOE1_Stationary", "PknIOE2_Stationary", "PknIOE3_Stationary",
          "PknJOE_3", "PknJOE1", "PknJOE1_Stationary", "PknJOE2", "PknJOE2_Stationary", "PknJOE3_Stationary",
          "PknK_OE_1", "PknK_OE_2", "PknK_OE_3",
          "PknLOE_1", "PknLOE_2", "PknLOE_3", "PknLOE_4", "PknLOE_5", "PknLOE_6", "PknLOE_Stat_4", "PknLOE_Stat_5", "PknLOE_Stat_6"],
    "KO":["PknB_KD_1", "PknB_KD_2", "PknB_KD_3", "PknBKD_1", "PknBKD_2", "PknBKD_3",
          "PknAKD_1", "PknAKD_2", "PknAKD_3",
          "PknD_KO_1", "PknD_KO_2", "PknD_KO_3", "PknDKO_Stat_4", "PknDKO_Stat_6",
          "PknEKO_1", "PknEKO_2", "PknEKO_3", "PknEKO_Expo_1", "PknEKO_Expo_2", "PknEKO_Expo_3",
          "PknFKO1_Stationary", "PknFKO2_Stationary", "PknFKO3_Stationary",
          "PknGKO1_Stationary", "PknGKO2_Stationary", "PknGKO3_Stationary",
          "PknH_KO_1", "PknH_KO_2", "PknH_KO_3",
          "PknIKO1_Stationary", "PknIKO2_Stationary", "PknIKO3_Stationary",
          "PknJKO1_Stationary", "PknJKO2_Stationary", "PknJKO3_Stationary",
          "PknK_KO_1", "PknK_KO_2", "PknK_KO_3",
          "PknLKO1_Stationary", "PknLKO2_Stationary", "PknLKO3_Stationary"]
}
oe_seq = kinase_rna_seq.loc[differential_kinase_conditions["OE"]].div(reference_condition_expression["OE"], axis=1)
ko_seq = kinase_rna_seq.loc[differential_kinase_conditions["KO"]].div(reference_condition_expression["KO"], axis=1)
kinase_rna_compendia = np.log2(pd.concat([oe_seq,ko_seq]))


# In[28]:


# Find the set of genes shared by both datasets, and remove genes that are not found in both
shared_genes = lh.find_intersect(list(filtered_rna_compendia.columns), list(kinase_rna_compendia.columns))
filtered_rna_compendia = filtered_rna_compendia[shared_genes]
kinase_rna_compendia = kinase_rna_compendia[shared_genes]


# ## Split Data into a Development set, and a testing set

# In[29]:


dev, test = sklearn.model_selection.train_test_split(filtered_rna_compendia, test_size=0.1, random_state=42)


# ## Hyperparameter Optimization for Elastic Net Regression

# In[30]:


# Convert the kinase targets dict to be by locus tag instead of kinase name
kinase_dict = {}
for key, item in kinase_targets_dict.items():
    kinase_dict[kinase_locus_dict[key]] = item
# Create the regression object
model = mycokinaseseq.kinase_tf_model.KinaseTfModel(kinase_dict=kinase_dict, tf_dict=tf_targets_dict, gene_list=list(dev.columns))

# Create lists for splitting the dataframes into kinase, tf, and other gene expression
kinase_list = list(model.kinase_dict.keys())
tf_list = list(model.tf_dict.keys())
targeted_gene_list = model.targeted_genes


# In[31]:


# Create the objective for the optuna Tree Parzen Estimator
def objective_cv(trial):
    # Define hyperparameter space
    alpha = trial.suggest_float("alpha",0.0,2.0)
    # For the L1_wt, it is suggested in sklearn to use more values closer to 1 to test with,
    # in order to do this, a categorical will be used with predefined values
    L1_wt = trial.suggest_float("L1_wt",0.0,1.0)

    # If the model is to be regularized, construct a kwarg dictionary
    kwarg_dict = {"alpha":alpha, "L1_wt":L1_wt}

    # Generate the cross validation iterator
    kfold = sklearn.model_selection.KFold(n_splits=5)

    # Separate out the dev set based on the kfold
    score_list = []
    for i, (train_index, test_index) in enumerate(kfold.split(dev)):
        model.fit(dev.iloc[train_index],significance_level=0.05,
                  multi_comparison_method="bh", false_discovery_rate=0.05,
                  regularized=True, verbose=True,
                  **kwarg_dict)
        # Add the score for each fold to the list
        score_list.append(
            model.score(kinase_expression=dev.iloc[test_index][kinase_list],
                        tf_expression=dev.iloc[test_index][tf_list],
                        gene_expression=dev.iloc[test_index][targeted_gene_list],
                        metric=sklearn.metrics.mean_squared_error).mean())
    score = np.mean(score_list)
    return score



def objective_single(trial):
    # Define hyperparameter space
    alpha = trial.suggest_float("alpha", 0.0,2.0)
    L1_wt = trial.suggest_float("L1_wt", 0.0,1.0)

    # Split the data to perform the scoring
    dev_train, dev_test = sklearn.model_selection.train_test_split(dev,test_size=0.2)

    # If the model is to be regularized, construct a kwarg dictionary
    kwarg_dict = {"alpha":alpha, "L1_wt":L1_wt}
    model.fit(dev_train,significance_level=0.05,
              multi_comparison_method="bh", false_discovery_rate=0.05,
              regularized=True, **kwarg_dict)
    return model.score(kinase_expression=dev_test[kinase_list],
                tf_expression=dev_test[tf_list],
                gene_expression=dev_test[targeted_gene_list],
                metric=sklearn.metrics.mean_squared_error).mean()


# In[ ]:


# If OPT_HYPERPARAMETERS is True, then run the optuna study, otherwise read the pre-generated values
if OPT_HYPERPARAMETERS:
    # Create the optuna study to find the optimum hyperparameters
    study = optuna.create_study(sampler=optuna.samplers.TPESampler(n_startup_trials=15, multivariate=True), direction="minimize")
    study.optimize(objective_cv, n_trials=200)
    # Get the results as a dataframe
    results = study.trials_dataframe()
else:
    with open(os.path.join(".", "optimum_hyperparamters.json"),"r") as f:
        hyperparameters_dict = json.load(f)


# In[ ]:


if OPT_HYPERPARAMETERS:
    sb.regplot(data = results, x= "params_alpha", y="value", lowess=True)


# In[ ]:


if OPT_HYPERPARAMETERS:
    sb.regplot(data = results, x= "params_L1_wt", y="value", lowess=True)


# In[ ]:


if OPT_HYPERPARAMETERS:
    fig = plt.figure()
    ax = plt.axes()
    sb.scatterplot(data=results, x="params_alpha", y="value", hue="params_L1_wt", palette="deep", ax=ax)


# In[ ]:


if OPT_HYPERPARAMETERS:
    opt_alpha = results.sort_values(by="value")["params_alpha"].iloc[0]
    opt_L1_wt = results.sort_values(by="value")["params_L1_wt"].iloc[0]
    print(f"The optimum value for the parameters found by the Tree Parzen estimator are:\n"
          f"alpha: {opt_alpha}\n"
          f"L1_wt: {opt_L1_wt}\n")
    # Save the optimum hyperparameters to avoid rerunning the hyperparameter optimization
    with open(os.path.join(".","optimum_hyperparamters.json",),"w") as f:
        json.dump({"opt_alpha":opt_alpha, "opt_L1_wt":opt_L1_wt}, f)
else:
    opt_alpha = hyperparameters_dict["opt_alpha"]
    opt_L1_wt = hyperparameters_dict["opt_L1_wt"]


# ## More Dev set testing

# In[ ]:


# Check if directory for this exists, and if not create it
cv_directory = os.path.join("..","data","cross_validation")
if not os.path.exists(cv_directory):
    os.mkdir(cv_directory)
#dev set testing
kfold = sklearn.model_selection.KFold(n_splits=5)
for index, (train_index, test_index) in enumerate(kfold.split(dev)):
    scores_outfile = os.path.join(cv_directory, f"fold{index}_scores.csv")
    model.fit(dev.iloc[train_index])
    score_df = model.score_full(kinase_expression=dev.iloc[test_index][kinase_list],
                                tf_expression=dev.iloc[test_index][tf_list],
                                gene_expression=dev.iloc[test_index][targeted_gene_list],
                                metric_list=[sklearn.metrics.mean_squared_error,
                                  sklearn.metrics.mean_absolute_error,
                                  sklearn.metrics.median_absolute_error,
                                  sklearn.metrics.r2_score],
                                metric_names=["mse","mae","median_abs_err","r2"],
                                cutoff=1.)
    score_df.to_csv(scores_outfile)


# ## Fit Model with Optimum Hyperparameters

# In[ ]:


model.fit(dev, significance_level=0.05, multi_comparison_method="bh", false_discovery_rate=0.05,regularized=True,
          intercept=True,verbose=True, alpha = opt_alpha, L1_wt = opt_L1_wt)


# ## Combine associations from the Model with Essentiality information

# In[ ]:


# Read in Essentiality Dataframe
# from Barbara Bosch, Michael A. DeJesus, Nicholas C. Poulton, Wenzhu Zhang, Curtis A. Engelhart, Anisha Zaveri, Sophie Lavalette,
# Nadine Ruecker, Carolina Trujillo, Joshua B. Wallach, Shuqi Li, Sabine Ehrt, Brian T. Chait, Dirk Schnappinger, Jeremy M. Rock,
# Genome-wide gene expression tuning reveals diverse vulnerabilities of M. tuberculosis, Cell, Volume 184, Issue 17, 2021,
# Pages 4579-4592.e24, ISSN 0092-8674, https://doi.org/10.1016/j.cell.2021.06.033. (https://www.sciencedirect.com/science/article/pii/S0092867421008242)
essentiality_df = pd.read_excel(os.path.join("..","data","Vulnerability_Index.xlsx"),
                             sheet_name="(1) Mtb H37Rv",
                             header=0)[["locus_tag","name","tnseq_ess","crispr_ess","Vulnerability Index"]]
 # Rename the locus tag to Rv.. form
essentiality_df["locus_tag"] = essentiality_df["locus_tag"].apply(lambda x: x.replace("RVBD","Rv"))
essentiality_df["name"] = essentiality_df["name"].apply(lambda x: x.replace("RVBD","Rv"))
# Rename locus_tag column to gene to match with associations dataframe
essentiality_df = essentiality_df.rename({"locus_tag":"gene"}, axis=1)
# Convert "gene" to be type string
essentiality_df["gene"] = essentiality_df["gene"].astype("string")


# In[ ]:


# Get associations dataframe from the fitted model
associations_df = model.create_associations_information_df()
# Convert "gene" column to be of type string
associations_df["gene"] = associations_df["gene"].astype("string")
# Merge the associations with the essentiality information
associations_df = associations_df.merge(essentiality_df, on="gene",how="left")


# In[ ]:


# Save the dataframe
associations_df.to_csv(os.path.join("..","output","associations.csv"))


# ## Analyze Model

# In[ ]:


# score the model against the test set from the original compendia
test_set_scores = model.score_full(kinase_expression=test[kinase_list], tf_expression=test[tf_list], gene_expression=test[targeted_gene_list])
test_set_scores.to_csv(os.path.join("..","data","output","test_set_scores.csv"))
# score the model against the independent kinase seq data
kinase_seq_scores = model.score_full(kinase_expression=kinase_rna_seq[kinase_list],
                                     tf_expression=kinase_rna_seq[tf_list],
                                     gene_expression=kinase_rna_seq[targeted_gene_list])
kinase_seq_scores.to_csv(os.path.join("..","data","output","kinase_seq_scores.csv"))


# ## Visualizations

# In[ ]:


essentiality_df = essentiality_df.set_index("gene")
network = model.create_network_from_associations(essentiality_df=essentiality_df, essentiality_col="crispr_ess")
with (os.path.join("..","data","output","associations_network.json"), "w") as f:
    json.dump(nx.node_link_data(network), f)
model.create_cytoscape_from_associations(essentiality_df=essentiality_df, essentiality_col="crispr_ess",
                                         outfile=os.path.join("..","data", "output","associations_network.cyjs"))

