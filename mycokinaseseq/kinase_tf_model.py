# Imports
import itertools
import typing
import warnings
from typing import Union, IO

import numpy as np
import pandas as pd
import sklearn
import statsmodels.api as sm

# Local Imports
import mycokinaseseq.list_helper as lh
from mycokinaseseq import progress_bar

# Stop warnings about chained assignment
pd.options.mode.chained_assignment = None


def multi_comparison_correction(data: pd.DataFrame,
                                method: str = "benjamani_hochberg",
                                p_val_col: str = "p_value",
                                significance_level: float = 0.05,
                                false_discovery_rate: float = 0.05) -> pd.DataFrame:
    """
    Perform multi comparison correction
    :param data: Data including the p_values to be corrected for multi comparison
    :param method: Method to use, can be None, Significance_Level, Bonferroni, or Benjamani_Hochberg
    :param p_val_col: Column of the data dataframe with the p-values
    :param significance_level: The significance level to use for the multi-comparison correction
    :param false_discovery_rate: The false discovery rate to use for the Benjamani Hochberg procedure
    :return: corrected_data
        Data corrected using the selected method
    """
    # If no correction desired, simple return the data
    # Only present for consistent syntax, doesn't actually do anything
    if not method:
        return data
    if method.upper() in ["NONE"]:
        return data
    # Return observations where the p-value is below the significance level
    if method.upper() in ["S", "SIGNIFICANCE", "SIGNIFICANCE LEVEL", "SIGNIFICANCE_LEVEL"]:
        return data.loc[data[p_val_col] < significance_level, :]
    # Return the observations which are below the significance level corrected by the Bonferroni method
    if method.upper() in ["B", "Bonferroni".upper()]:
        return data.loc[data[p_val_col] < significance_level / len(data), :]
    # Return the observations which are below the significance level corrected using the
    # Benjamani-Hochberg procedure
    if method.upper() in ["BH", "Benjamini-Hochberg".upper(), "Benjamini_Hochberg".upper(),
                          "Benjamini Hochberg".upper(), "FDR", "FALSE DISCOVERY RATE", "FALSE_DISCOVERY_RATE",
                          "FALSE-DISCOVERY-RATE"]:
        original_cols = list(data.columns)
        ranked_data = data.sort_values(by=[p_val_col], ignore_index=True, ascending=True)
        ranked_data["rank"] = range(1, len(ranked_data) + 1)
        number_tests = len(ranked_data)
        ranked_data["BH_critical_value"] = (ranked_data["rank"] / number_tests) * false_discovery_rate
        critical_p = ranked_data.loc[ranked_data[p_val_col] <
                                     ranked_data["BH_critical_value"]][p_val_col].max()
        corrected_data = ranked_data.loc[ranked_data[p_val_col] < critical_p]
        corrected_data = corrected_data[(original_cols + ["BH_critical_value"])]
        return corrected_data


# A class which can take in an RNA seq compendia in log2(fold-change) form,
# and create a set of linear models to predict gene expression when given kinase
# and transcription factor information
# The data will not be scaled

class KinaseTfModel:
    def __init__(self, kinase_dict: dict, tf_dict: dict, gene_list: list):
        """
        Initiation Method
        :param kinase_dict: Dictionary of kinase: gene-targets
        :param tf_dict: Dictionary of tf: gene-targets
        """
        # Create variables to store the kinase and TF dicts
        self.kinase_dict = kinase_dict
        self.tf_dict = tf_dict
        # Create a list of genes
        self.gene_list = gene_list
        # Create dictionary of tf:kinase
        self.tf_kinase_dict = {}
        # Iterate through the transcription factors
        for tf in self.tf_dict.keys():
            kinase_list = []
            # Iterate through the kinases
            for kinase in self.kinase_dict.keys():
                # If the tf is targeted by the kinase, add it to the
                if tf in self.kinase_dict[kinase]:
                    kinase_list.append(kinase)
            self.tf_kinase_dict[tf] = lh.get_unique_list(kinase_list)
        # Create dictionaries to hold information about TF and Kinase targets, keyed by gene
        self.gene_to_kinase_dict = {}  # gene: kinase which directly targets that gene
        self.gene_to_tf_dict = {}  # gene: tf which directly targets that gene
        self.gene_to_tf_to_kinase_dict = {}  # gene: kinase which targets that gene through tf
        # Iterate through the genes to construct the dictionaries
        for gene in self.gene_list:
            kinase_list = []
            tf_list = []
            tf_kinase_list = []
            for kinase in self.kinase_dict.keys():
                if gene in self.kinase_dict[kinase]:
                    kinase_list.append(kinase)
            for tf in self.tf_dict.keys():
                if gene in self.tf_dict[tf]:
                    tf_list.append(tf)
                    tf_kinase_list += self.tf_kinase_dict[tf]
            self.gene_to_tf_dict[gene] = lh.get_unique_list(tf_list)
            self.gene_to_kinase_dict[gene] = lh.get_unique_list(kinase_list)
            self.gene_to_tf_to_kinase_dict[gene] = lh.get_unique_list(tf_kinase_list)
        # Create lists of genes targeted by TFs, targeted by TFs targeted by kinases, and targeted by kinases
        self.genes_targeted_by_tf = []
        self.genes_targeted_by_kinase = []
        self.genes_targeted_by_tf_targeted_by_kinase = []
        # Iterate through the genes
        for gene in self.gene_list:
            # If the gene is targeted by TFs,
            if self.gene_to_tf_dict[gene]:
                self.genes_targeted_by_tf += self.gene_to_tf_dict[gene]
            if self.gene_to_kinase_dict[gene]:
                self.genes_targeted_by_kinase += self.gene_to_kinase_dict[gene]
            if self.gene_to_tf_to_kinase_dict[gene]:
                self.genes_targeted_by_tf_targeted_by_kinase += self.gene_to_tf_to_kinase_dict[gene]
        # Array to hold the model coefficients which will be determined during fitting
        self.model_coefficients = None
        # Array to hold the associations, again determined during fitting
        self.associations = None
        # Array to hold the significant associations, determined during fitting
        self.significant_associations = None
        # Variable for if the models have intercepts
        self.intercept = None

    def fit(self, compendia: pd.DataFrame,
            significance_level: float = 0.05,
            multi_comparison_method: str = "bh",
            false_discovery_rate: float = 0.05,
            regularized: bool = False,
            intercept: bool = True,
            verbose: bool = False,
            **kwargs
            ):
        """
        Method to fit the model to provided compendia
        :type kwargs: dict of key word arguments passed to the statsmodels regression fit
        :param regularized: Whether to use regularization when fitting the linear model
        :param compendia: RNA seq compendia, log2(fold-change) form. Genes (in Rv.. form) as columns, samples as rows
        :param significance_level: Significance level to use for finding associations
        :param multi_comparison_method: Method to use for multi-comparison correction
        :param false_discovery_rate: False discovery rate to use in Benjamani-Hochberg procedure if that is selected as
            the multi-comparison correction method
        :param intercept: Whether to include an intercept in the model
        :param verbose: Whether a verbose output is desired
        :return: None
        """
        # Add variable for if the model will have an intercept
        self.intercept = intercept
        # This will be in two stages, the first will find significant associations, the second will find the actual
        # coefficients
        # First, find the associations
        self.find_associations(compendia=compendia,
                               significance_level=significance_level,
                               multi_comparison_method=multi_comparison_method,
                               false_discovery_rate=false_discovery_rate,
                               intercept=intercept,
                               verbose=verbose)
        # Now fit the model again with the significant associations being used for the interaction terms,
        # with regularization if desired
        self.find_model_coefficients(compendia=compendia,
                                     intercept=intercept,
                                     verbose=verbose,
                                     regularized=regularized,
                                     **kwargs)

    def predict_single(self, tf_expression: pd.Series, kinase_expression: pd.Series):
        """
        Method to predict gene expression values based on provided TF and Kinase expression values
        :param tf_expression:
        :param kinase_expression:
        :return: Prediction
            Pandas Series of predictions, index is locus tag, value is the log2(fold-change) expression value
        """
        # Check if all the TFs are in the model, drop any that aren't
        if not set(tf_expression.index).issubset(set(self.model_coefficients.columns)):
            warnings.warn("Some TFs in provided expression not found in model, removing")
            # Remove all entries in tf_expression, not in model_coefficients
            remove_list = list(set(tf_expression.index) - set(self.model_coefficients.columns))
            # Drop them
            tf_expression = tf_expression.drop(remove_list)
        # Check if all the kinases are in the model, drop any that aren't
        if not set(kinase_expression.index).issubset(set(self.model_coefficients.columns)):
            warnings.warn("Some kinases in provided expression not found in model, removing")
            # Remove all entries in kinase_expression, not in model_coefficients
            remove_list = list(set(kinase_expression.index) - set(self.model_coefficients.columns))
            # Drop them
            tf_expression = kinase_expression.drop(remove_list)
        # Create the pandas series that will be used as input
        input_series = pd.Series(0, index=self.model_coefficients.columns)
        # Set the TF expression values
        input_series[tf_expression.index] = tf_expression
        # Set the Kinase expression values
        input_series[kinase_expression.index] = kinase_expression
        # Now add interaction terms
        for tf in self.tf_dict.keys():
            for kinase in self.kinase_dict.keys():
                if tf in self.kinase_dict[kinase]:
                    rep = self.interaction_term_repr((kinase, tf))
                    if rep in input_series.index:
                        input_series[rep] = float(tf_expression[tf] * tf_expression)
        # Add term for intercept if needed
        if self.intercept:
            input_series["intercept"] = 1.0
        prediction = pd.Series(np.matmul(self.model_coefficients.fillna(0).to_numpy(),
                                         input_series.to_numpy()).flatten(),
                               index=self.model_coefficients.index)
        return prediction

    def predict(self, tf_expression: pd.DataFrame, kinase_expression: pd.DataFrame):
        """
        Predict the gene expression values from
        :param tf_expression: Dataframe of TF expression, indexed by sample
        :param kinase_expression: Dataframe of Kinase expression, indexed by sample
        :return: prediction
            Dataframe of predicted gene expression, indexed by sample
        """
        # Check if all the TFs are in the model
        if not set(tf_expression.columns).issubset(set(self.model_coefficients.columns)):
            warnings.warn("Some TFs in provided expression not found in model, removing")
            # Remove all entries in tf_expression, not in model_coefficients
            remove_list = list(set(tf_expression.columns) - set(self.model_coefficients.columns))
            # Drop them
            tf_expression = tf_expression.drop(remove_list, axis=1)
        # Check if all the kinases are in the model, drop any that aren't
        if not set(kinase_expression.columns).issubset(set(self.model_coefficients.columns)):
            warnings.warn("Some kinases in provided expression not found in model, removing")
            # Remove all entries in kinase_expression, not in model_coefficients
            remove_list = list(set(kinase_expression.columns) - set(self.model_coefficients.columns))
            # Drop them
            tf_expression = kinase_expression.drop(remove_list, axis=1)
        # Ensure that tf_expression, and kinase_expression have the same index
        if not (tf_expression.index == kinase_expression.index).all():
            warnings.warn("Indices are not identical, finding overall, dropping remainder")
            # Find intersect
            intersect = lh.find_intersect(list(tf_expression.index), list(kinase_expression.index))
            tf_expression = tf_expression.loc[intersect]
            kinase_expression = kinase_expression.loc[intersect]
        # Create dataframe for input data
        input_data = pd.DataFrame(0.,
                                  index=tf_expression.index,
                                  columns=self.model_coefficients.columns)
        # Add the interaction terms
        for tf in self.tf_dict.keys():
            for kinase in self.kinase_dict.keys():
                if tf in self.kinase_dict[kinase]:
                    rep = f"{kinase}_{tf}"
                    if rep in input_data.columns:
                        input_data[rep] = input_data[kinase] * input_data[tf]
        if self.intercept:
            input_data["intercept"] = 1.0
        # predict with model
        prediction = pd.DataFrame(np.transpose(np.matmul(self.model_coefficients.fillna(0).to_numpy(),
                                                         np.transpose(input_data.to_numpy()))),
                                  index=kinase_expression.index, columns=self.model_coefficients.index)
        return prediction

    # The input dataframes should be indexed by sample
    def score(self, kinase_expression: pd.DataFrame,
              tf_expression: pd.DataFrame,
              gene_expression: pd.DataFrame,
              metric: typing.Callable) -> pd.DataFrame:
        """
        Score the model against the provided data
        :param metric: Metric to use for scoring
        :param kinase_expression: Dataframe of kinase expression values
        :param tf_expression: Dataframe of TF expression values
        :param gene_expression: Dataframe of gene expression values
        :return: Scores
            Dataframe of scores
        """
        pass

    def find_interaction_terms_initial(self, gene):
        """
        Find the interaction terms for the linear model from the target information
        :param gene:
        :return: interaction_terms
            List of (kinase, TF) tuples representing the interaction terms
        """
        # Find the tfs which target the gene
        tfs = self.gene_to_tf_dict[gene]
        # Now find the interaction terms
        # Create a list of tuples with (kinase, tf) where the kinase targets the TF, and the TF targets the genes
        interaction_terms = []
        for tf in tfs:
            # Check if there are kinases which target this tf
            if self.tf_kinase_dict[tf]:
                # Add interaction terms to list
                for kinase in self.tf_kinase_dict[tf]:
                    interaction_terms.append((kinase, tf))
            else:
                continue
        return interaction_terms

    @staticmethod
    def find_interaction_terms_from_associations(gene,
                                                 significant_associations,
                                                 gene_col: str = "gene",
                                                 tf_col: str = "TF",
                                                 kinase_col: str = "kinase"):
        """
        Find the interaction terms
        :param kinase_col: Column with the kinase locus tag
        :param tf_col: Columns with the TF locus tag
        :param gene_col: Columns with the gene locus tag
        :param significant_associations: The significant associations to use
        :param gene: Gene to find the interaction terms for
        :return: interaction_terms
            List of (kinase, TF) tuples where the
        """
        # Filter the dataframe for only those columns
        filtered_associations = significant_associations.loc[
            significant_associations[gene_col] == gene][[kinase_col, tf_col]]
        return list(filtered_associations.itertuples(index=False, name=None))

    def create_exogenous_array(self,
                               compendia,
                               gene,
                               interaction_terms: list[tuple],
                               intercept: bool = True) -> pd.DataFrame:
        """
        Create exogenous matrix for statsmodels OLS model fitting and association evaluation
        :param interaction_terms: Interaction terms for creating the exogenous array
        :param intercept: Whether an intercept term should be added
        :param gene: gene for which model is being created
        :param compendia: RNA seq compendia, log2(fold-change) form. Genes as columns, samples as rows
        :return: (interaction_terms, exogenous_array) tuple
            interaction_terms-list of interaction term tuples
            exogenous_array-Exogenous array for the statsmodels OLS fitting
        """
        tfs = self.gene_to_tf_dict[gene]
        # Check if there are kinases which target tfs which target this gene
        if self.gene_to_tf_to_kinase_dict[gene]:
            kinases = self.gene_to_tf_to_kinase_dict[gene]
        else:
            kinases = []
        # Create the exogenous array with the transcription factors and kinases
        exogenous_array = compendia[tfs + kinases]
        for kinase, tf in interaction_terms:
            str_rep = self.interaction_term_repr((kinase, tf))
            exogenous_array[str_rep] = exogenous_array[kinase] * exogenous_array[tf]
        if intercept:
            exogenous_array["intercept"] = 1.
        return exogenous_array

    @staticmethod
    def interaction_term_repr(term: tuple[str, str]):
        return f"{term[0]}_{term[1]}"

    @staticmethod
    def interaction_term_repr_list(term_list: list[tuple[str, str]]):
        repr_list = []
        for term in term_list:
            repr_list.append(KinaseTfModel.interaction_term_repr(term))
        return repr_list

    def find_associations(self,
                          compendia: pd.DataFrame,
                          significance_level: float = 0.05,
                          multi_comparison_method: str = "bh",
                          false_discovery_rate: float = 0.05,
                          intercept: bool = False,
                          verbose: bool = False):
        """
        Method to find the significant associations
        :param intercept: Whether an intercept term should be added
        :param compendia: RNA seq compendia, log2(fold-change) form, columns are genes, rows are samples
        :param verbose: Whether a verbose output is desired
        :param significance_level: Significance level
        :param multi_comparison_method: Method to use for multi-comparison correction
        :param false_discovery_rate: False discovery rate to use for BH procedure if that is selected
        :return: associations
            Dataframe with associations
        """
        # Create a list of genes which are targeted by at least 1 TF (otherwise, the model will have no information)
        targeted_genes_list = list(filter(lambda x: ((x not in list(self.tf_dict.keys())) and
                                                     (x not in list(self.kinase_dict.keys()))),
                                          self.genes_targeted_by_tf))
        if verbose:
            print("Finding Significant Associations")
            bar = progress_bar.ProgressBar(total=len(targeted_genes_list), divisions=10)
        associations_dict = {
            "kinase": [],
            "TF": [],
            "gene": [],
            "p_value": [],
            "r_squared": [],
            "adjusted_r_squared": [],
            "tf_coef": [],
            "kinase_coef": [],
            "interaction_coef": []
        }
        associations_df_list = []
        for gene in targeted_genes_list:
            if verbose:
                # noinspection PyUnboundLocalVariable
                bar.inc()
            data = compendia.copy()
            interaction_terms = self.find_interaction_terms_initial(gene)
            exogenous_array = self.create_exogenous_array(compendia=compendia,
                                                          gene=gene,
                                                          interaction_terms=interaction_terms,
                                                          intercept=intercept).dropna(axis=0)
            endogenous_array = data[gene].dropna(axis=0)
            defined_samples = lh.find_intersect(list(exogenous_array.index), list(endogenous_array.index))
            exogenous_array = exogenous_array.loc[defined_samples]
            endogenous_array = endogenous_array.loc[defined_samples]
            model = sm.OLS(endog=endogenous_array, exog=exogenous_array)
            results = model.fit()
            for kinase, tf in interaction_terms:
                representation = f"{kinase}_{tf}"
                associations_dict["kinase"].append(kinase)
                associations_dict["TF"].append(tf),
                associations_dict["gene"].append(gene),
                associations_dict["p_value"].append(results.pvalues[representation]),
                associations_dict["r_squared"].append(results.rsquared),
                associations_dict["adjusted_r_squared"].append(results.rsquared_adj),
                associations_dict["tf_coef"].append(results.params[tf]),
                associations_dict["kinase_coef"].append(results.params[tf]),
                associations_dict["interaction_coef"].append(results.params[representation])
            associations_df_list.append(pd.DataFrame(associations_dict))
        self.associations = pd.concat(associations_df_list, axis=0, ignore_index=True)
        self.associations.drop_duplicates(inplace=True)
        if verbose:
            print("Found Associations")
        self.significant_associations = multi_comparison_correction(self.associations,
                                                                    method=multi_comparison_method,
                                                                    p_val_col="p_value",
                                                                    significance_level=significance_level,
                                                                    false_discovery_rate=false_discovery_rate)

    def find_model_coefficients(self,
                                compendia: pd.DataFrame,
                                intercept: bool = True,
                                verbose: bool = False,
                                regularized: bool = False,
                                **kwargs):
        """
        Method to find the model coefficients based on significant associations
        :param regularized: Whether statsmodels linear regression should be regularized
        :param compendia: RNA seq compendia for fitting the model, in log2(fold-change) form, with genes as columns
            and samples as rows
        :param intercept: Whether an intercept should be added to the model
        :param verbose: Whether a verbose output is desired
        :param kwargs: Dict of keyword args to pass to statsmodels fit method
        :return: None
        """
        # Create a list of genes which are targeted by at least 1 TF (otherwise, the model will have no information)
        targeted_genes_list = list(filter(lambda x: ((x not in list(self.tf_dict.keys())) and
                                                     (x not in list(self.kinase_dict.keys()))),
                                          self.genes_targeted_by_tf))
        if verbose:
            print("Finding Model Coefficients")
            bar = progress_bar.ProgressBar(total=len(targeted_genes_list), divisions=10)
        all_interaction_terms_list = []
        for tf in self.tf_dict.keys():
            kinases = self.tf_kinase_dict[tf]
            for kinase in kinases:
                all_interaction_terms_list.append(f"{kinase}_{tf}")
        self.model_coefficients = pd.DataFrame(data=0.,
                                               index=targeted_genes_list,
                                               columns=list(self.tf_dict.keys()) + list(self.kinase_dict.keys()) +
                                                       all_interaction_terms_list)
        self.model_coefficients["intercept"]=0.0
        for gene in targeted_genes_list:
            if verbose:
                # noinspection PyUnboundLocalVariable
                bar.inc()
            tfs = self.gene_to_tf_dict[gene]
            kinases = self.gene_to_tf_to_kinase_dict[gene]
            data = compendia.copy()
            interaction_terms = self.find_interaction_terms_from_associations(gene,
                                                                              self.significant_associations,
                                                                              gene_col="gene",
                                                                              tf_col="TF",
                                                                              kinase_col="kinase")
            interaction_term_list = self.interaction_term_repr_list(interaction_terms)
            exogenous_array = self.create_exogenous_array(compendia=compendia,
                                                          gene=gene,
                                                          interaction_terms=interaction_terms,
                                                          intercept=intercept).dropna(axis=0)
            endogenous_array = data[gene].dropna(axis=0)
            defined_samples = lh.find_intersect(list(exogenous_array.index), list(endogenous_array.index))
            exogenous_array = exogenous_array.loc[defined_samples]
            endogenous_array = endogenous_array.loc[defined_samples]
            model = sm.OLS(endog=endogenous_array, exog=exogenous_array)
            if regularized:
                results = model.fit_regularized(**kwargs)
            else:
                results = model.fit(**kwargs)
            coefficients = results.params
            for tf in tfs:
                if tf in coefficients.index:
                    self.model_coefficients.at[gene, tf] = coefficients[tf]
            for kinase in kinases:
                if kinase in coefficients.index:
                    self.model_coefficients.at[gene, kinase] = coefficients[kinase]
            for interaction_term in interaction_term_list:
                if interaction_term in coefficients.index:
                    self.model_coefficients.at[gene, interaction_term] = coefficients[interaction_term]
            if intercept:
                self.model_coefficients.at[gene, "intercept"] = coefficients["intercept"]
        if verbose:
            print("Found Coefficients")
            print("Finding kinase effect on genes")

    def create_associations_information_df(self) -> pd.DataFrame:
        """
        Add additional information about number of TF and Kinase targets, along with
        :return: associations
            Dataframe with associations and added information
        """
        # Get a copy of the significant associations
        associations = self.significant_associations.copy()
        # Add information on the effect of
        associations["effect_on_tf"] = np.sign(associations["tf_coef"]) * np.sign(associations["interaction_coef"])
        associations["effect_on_gene"] = np.sign(associations["tf_coef"]) * associations["effect_on_tf"]
        associations["tfs_targeting_gene"] = associations["gene"].apply(
            lambda g: len(lh.get_unique_list(self.gene_to_tf_dict[g])))
        associations["kinases_targeting_tf"] = associations["tf"].apply(
            lambda t: len(lh.get_unique_list(self.tf_kinase_dict[t])))
        associations["kinases_targeting_tfs_targeting_gene"] = associations["gene"].apply(
            lambda g: len(lh.get_unique_list(self.gene_to_tf_to_kinase_dict[g])))
        associations["tf_targets_num_genes"] = associations["TF"].apply(
            lambda t: len(lh.get_unique_list(self.tf_dict[t])))
        associations["kinase_targets_num_genes"] = associations["kinase"].apply(
            lambda k: len(
                lh.get_unique_list(
                    lh.find_intersect(
                        self.kinase_dict[k],
                        list(self.tf_dict.keys())))))
        return associations

    def save_associations(self, file_path: Union[IO, str]) -> None:
        """
        Save the associations and relevant information to a csv
        :param file_path: Where to save the csv to
        :return: None
        """
        # get a copy of the significant associations
        associations = self.create_associations_information_df()
        # Save the dataframe
        associations.to_csv(file_path)
