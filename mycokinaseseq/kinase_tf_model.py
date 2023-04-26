# Imports
import itertools
import json
import os
import typing
import warnings
from typing import Union, IO

# External library imports
import networkx as nx
import numpy as np
import pandas as pd
import scipy.stats
from scipy.stats import gaussian_kde
import sklearn.metrics
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score, median_absolute_error, roc_auc_score
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
        :param gene_list: List of all genes in the compendia
        """
        # Create a list of genes
        self.gene_list = gene_list
        # Remove any genes not in gene_list from both of the dictionaries
        filtered_kinase_dict = {}
        filtered_tf_dict = {}
        for kinase, gene_list in kinase_dict.items():
            if kinase not in self.gene_list:
                continue
            filtered_genes = [gene for gene in gene_list if gene in self.gene_list]
            filtered_kinase_dict[kinase] = filtered_genes
        for tf, gene_list in tf_dict.items():
            if tf not in self.gene_list:
                continue
            filtered_genes = [gene for gene in gene_list if gene in self.gene_list]
            filtered_tf_dict[tf] = filtered_genes
        # Create variables to store the kinase and TF dicts
        self.kinase_dict = filtered_kinase_dict
        self.tf_dict = filtered_tf_dict

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
        # Create a list of all the genes which are targeted by at least one tf, which is targeted by a kinase
        self.targeted_genes = []
        for key, item in self.gene_to_tf_to_kinase_dict.items():
            if item:
                if (key not in list(self.tf_dict.keys())) and (key not in list(self.kinase_dict.keys())):
                    self.targeted_genes.append(key)
        # Lists of TFs and Kinases
        self.tf_list = list(tf_dict.keys())
        self.kinase_list = list(kinase_dict.keys())
        # Array to hold the model coefficients which will be determined during fitting
        self.model_coefficients = None
        # Array to hold the associations, again determined during fitting
        self.associations = None
        # Array to hold the significant associations, determined during fitting
        self.significant_associations = None
        # Variable for if the models have intercepts
        self.intercept = None
        # Variable for if the model is regularized
        self.regularized = None

    def fit(self, compendia: pd.DataFrame,
            significance_level: float = 0.05,
            multi_comparison_method: str = "bh",
            false_discovery_rate: float = 0.05,
            regularized: bool = True,
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
        self.regularized = regularized
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
        # Create a blank input dataframe
        input_data = pd.DataFrame(0., columns=self.model_coefficients.columns, index=tf_expression.index)
        # Add the tf expression data to the input data
        input_data.loc[tf_expression.index, tf_expression.columns] = tf_expression
        # Add the kinase expression data to the input data
        input_data.loc[kinase_expression.index, kinase_expression.columns] = kinase_expression
        # Add in the intercept term if needed
        if self.intercept:
            input_data["intercept"] = 1.0
        # Add in the interaction terms
        for tf in self.tf_kinase_dict.keys():
            for kinase in self.tf_kinase_dict[tf]:
                representation = self.interaction_term_repr((kinase, tf))
                input_data[representation] = input_data[kinase] * input_data[tf]
        # Reorder the input data to make sure it matches the model_coefficients
        input_data = input_data[self.model_coefficients.columns]
        # predict with model
        prediction = pd.DataFrame(np.transpose(np.matmul(self.model_coefficients.fillna(0).to_numpy(),
                                                         np.transpose(input_data.to_numpy()))),
                                  index=kinase_expression.index, columns=self.model_coefficients.index)
        return prediction

    # The input dataframes should be indexed by sample
    def score(self, kinase_expression: pd.DataFrame,
              tf_expression: pd.DataFrame,
              gene_expression: pd.DataFrame,
              metric: typing.Callable = mean_squared_error) -> pd.Series:
        """
        Score the model against the provided data
        :param metric: Metric to use for scoring, function which can take in two Series of values,
            the first being the true values, the second being the predicted and return a numerical score
        :param kinase_expression: Dataframe of kinase expression values
        :param tf_expression: Dataframe of TF expression values
        :param gene_expression: Dataframe of gene expression values
        :return: score_series
            Pandas series of scores
        """
        predicted_gene_expression = self.predict(tf_expression=tf_expression, kinase_expression=kinase_expression)
        score_series = pd.Series(index=list(predicted_gene_expression.columns), dtype="Float64")
        for gene in gene_expression.columns:
            predicted = predicted_gene_expression[gene]
            true = gene_expression[gene]
            score_series[gene] = metric(true, predicted)
        return score_series

    def score_classification(self,
                             kinase_expression: pd.DataFrame,
                             tf_expression: pd.DataFrame,
                             gene_expression: pd.DataFrame,
                             cutoff: float = 1.,
                             cutoff_method = "simple") -> (pd.Series, pd.Series):
        """
        Score the model against the gene expression data, based on classification of gene expression as either
            over expression, or under expression
        :param cutoff_method: How to determine which genes are differentially expressed,
            can be "simple" or "prob"
            simple uses cutoff as a cutoff value
            prob uses cutoff as a probability (or proportion)
        :param kinase_expression: Dataframe of kinase expression values
        :param tf_expression: Dataframe of TF expression values
        :param gene_expression: Dataframe of gene expression values
        :return: score_series
            Pandas series of scores
        :param cutoff: Cutoff for over, and under expression
            if cutoff_method is simple: under expression is defined as less than the negative
                cutoff, and over expression is defined as greater than the cutoff
            if cutoff_method is prob: under expression is defined as being in the bottom cutoff-proportion of the data,
            and over expression is defined as being the top cutoff-proportion of the data
        :return: (oe_score, ue_scores)
            Tuple of pandas series containing the roc_auc scores for the classification problem
        """
        if cutoff_method.upper() in ["SIMPLE", "S"]:
            # Binarize the gene expression data, cutting off at the cutoff parameter
            # Into two dataframes, one for over expression, and one for under expression
            oe_df = (gene_expression > cutoff)
            ko_df = (gene_expression < -1 * cutoff)
        # Convert the gene_expression into a probability dataframe, then use the cutoff proportion to find
        #   the over and under expressed genes
        if cutoff_method.upper() in ["PROB", "PROP", "PROBABILITY", "PROPORTION"]:
            prob_df_true = pd.DataFrame(0, index = gene_expression.index, columns=gene_expression.columns)
            for gene in prob_df_true.columns:
                prob_df_true[gene] = self.compute_probabilities(gene_expression[gene], "left")
            oe_df = (prob_df_true > (1-cutoff))
            ko_df = (prob_df_true < cutoff)
        # Create the predicted value dataframe
        predicted_df = self.predict(tf_expression=tf_expression, kinase_expression=kinase_expression)
        # Transform the predicted value dataframe,
        #       First, compute a gaussian kernel for each gene's predictions
        #       Then calculate the CDF up to that value to get a p value for the over expression case
        #       Invert this (take 1-value) to get the under expression predictions
        oe_prob_df = pd.DataFrame(0, index=predicted_df.index, columns=predicted_df.columns)
        for gene in predicted_df.columns:
            oe_prob_df[gene] = self.compute_probabilities(predicted_df[gene], "left")
        ko_prob_df = 1-oe_prob_df
        # Use sklearn multilabel roc_auc to find the auc value for each gene, for both OE, and UE
        # Won't work for the cases where there are no examples of overexpression,
        oe_sum = oe_df.sum()
        # Find genes where there are no examples of overexpression
        no_oe_genes = list(oe_sum[oe_sum == 0].index)
        # Find the genes where there are examples of overexpression
        oe_genes = list(oe_sum[oe_sum != 0].index)
        # Score the genes that have at least 1 case of overexpression
        oe_score = pd.Series(roc_auc_score(oe_df[oe_genes], oe_prob_df[oe_genes], average=None), index=oe_genes)
        # Add Nones for all the genes that can't be scored
        no_oe_score = pd.Series(None, index=no_oe_genes, dtype="Float64")
        # Concatenate the two series to create the final score series
        oe_score = pd.concat([oe_score, no_oe_score])
        # Name the series for later concatenation into a dataframe (if using full_score method)
        oe_score.name = "oe_auc_score"
        # Find the sum of the ko_df to find cases where there are no examples of under expression, (sum=0)
        ko_sum = ko_df.sum()
        # Find all genes where there are no examples of under expression
        no_ko_genes = list(ko_sum[ko_sum == 0].index)
        # Find all genes where there are examples of under expression
        ko_genes = list(ko_sum[ko_sum!=0].index)
        # Compute the ko_score for the genes where that is defined
        ko_score = pd.Series(roc_auc_score(ko_df[ko_genes], ko_prob_df[ko_genes], average=None), index=ko_genes)
        # Add none for all the genes that can't be scored
        no_ko_score = pd.Series(None, index=no_ko_genes, dtype="Float64")
        # Concatenate the two series to create the final score series
        ko_score = pd.concat([ko_score, no_ko_score])
        # Name the series for the later concatenation into a dataframe (if using full_score method)
        ko_score.name = "ue_auc_score"
        # Gather data into dataframe of scores
        return oe_score, ko_score

    def score_full(self, kinase_expression: pd.DataFrame,
                   tf_expression: pd.DataFrame,
                   gene_expression: pd.DataFrame,
                   metric_list=None,
                   metric_names=None,
                   cutoff: float = 1.) -> pd.DataFrame:
        """
        Function to score the model, using both regression scores and classification scores
        :param kinase_expression: Kinase expression values, shape is (samples, kinases)
        :param tf_expression: TF expression values, shape is (samples, tfs)
        :param gene_expression: Gene expression values to score against, shape is (samples, genes)
        :param metric_list: list of metrics for the regression scoring
        :param metric_names: list of names of the metrics provided in metric list
        :param cutoff: Cutoff for labelling the gene_expression data for classification scoring
        :return: score_df
            Pandas dataframe of scores, shape is (genes, scores)
        """
        if metric_list is None:
            metric_list = [mean_squared_error,
                           mean_absolute_error,
                           r2_score,
                           median_absolute_error]
            metric_names = ["mean_squared_error",
                            "mean_absolute_error",
                            "median_absolute_error",
                            "r2_score"]
        if len(metric_list) != len(metric_names):
            raise ValueError("Length of metric_list and metric_names don't match")
        scores = []
        for metric, name in zip(metric_list, metric_names):
            score = self.score(kinase_expression=kinase_expression,
                               tf_expression=tf_expression,
                               gene_expression=gene_expression,
                               metric=metric)
            score.name = name
            scores.append(score)
        oe_scores, ue_scores = self.score_classification(kinase_expression=kinase_expression,
                                                         tf_expression=tf_expression,
                                                         gene_expression=gene_expression,
                                                         cutoff=cutoff)
        scores.append(oe_scores)
        scores.append(ue_scores)
        return pd.concat(scores, axis=1)

    def find_interaction_terms_initial(self, gene):
        """
        Find the interaction terms for the linear model from the target information
        :param gene: Gene to find the interaction terms for
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
        if verbose:
            print("Finding Significant Associations")
            bar = progress_bar.ProgressBar(total=len(self.targeted_genes), divisions=10)
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
        for gene in self.targeted_genes:
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
                associations_dict["kinase_coef"].append(results.params[kinase]),
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
                                regularized: bool = True,
                                **kwargs):
        """
        Method to find the model coefficients based on significant associations.
        NOTE: Can't use refit, or an L1_wt of 0, because both will lead to the coefficients being returned as a ndarray
        and what TFs and interactions terms the fitted values correspond too can't be recovered
        :param regularized: Whether statsmodels linear regression should be regularized
        :param compendia: RNA seq compendia for fitting the model, in log2(fold-change) form, with genes as columns
            and samples as rows
        :param intercept: Whether an intercept should be added to the model
        :param verbose: Whether a verbose output is desired
        :param kwargs: Dict of keyword args to pass to statsmodels fit method
        :return: None
        """
        # Save whether the model is regularized
        self.regularized = regularized
        # Check for L1_wt, and refit
        if "refit" in kwargs.keys():
            if kwargs["refit"]:
                raise ValueError("refit can't be used due to issues with statsmodels return type")
        if "L1_wt" in kwargs.keys():
            if kwargs["L1_wt"] == 0 or kwargs["L1_wt"] == 0.:
                raise ValueError("L1_wt can't be zero due to issues with statsmodels return type")
        if verbose:
            print("Finding Model Coefficients")
            bar = progress_bar.ProgressBar(total=len(self.targeted_genes), divisions=10)
        all_interaction_terms_list = []
        for tf in self.tf_dict.keys():
            kinases = self.tf_kinase_dict[tf]
            for kinase in kinases:
                all_interaction_terms_list.append(f"{kinase}_{tf}")
        self.model_coefficients = pd.DataFrame(data=0.,
                                               index=self.targeted_genes,
                                               columns=list(self.tf_dict.keys()) + list(self.kinase_dict.keys()) +
                                                       all_interaction_terms_list)
        if intercept:
            self.model_coefficients["intercept"] = 0.0
        for gene in self.targeted_genes:
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
            # TODO: Change how the model deals with NA entries
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
        associations["kinases_targeting_tf"] = associations["TF"].apply(
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
        # If the model is regularized, add the regularized coefficients as well to the associations dataframe
        if self.regularized:
            # Add columns to the dataframe to hold the data
            associations["tf_coefficient_regularized"] = 0.
            associations["kinase_coefficient_regularized"] = 0.
            associations["interaction_coefficient_regularized"] = 0.
            # Iterate through the rows to find entry in the coefficient matrix and add the value to the row
            for index, series in associations.iterrows():
                gene = series["gene"]
                tf = series["TF"]
                kinase = series["kinase"]
                interaction_term = self.interaction_term_repr((kinase, tf))
                tf_coefficient = self.model_coefficients.loc[gene, tf]
                kinase_coefficient = self.model_coefficients.loc[gene, kinase]
                interaction_coefficient = self.model_coefficients.loc[gene, interaction_term]
                associations.loc[index, "tf_coefficient_regularized"] = tf_coefficient
                associations.loc[index, "kinase_coefficient_regularized"] = kinase_coefficient
                associations.loc[index, "interaction_coefficient_regularized"] = interaction_coefficient
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

    def create_network_from_associations(self,
                                         essentiality_df: pd.DataFrame,
                                         essentiality_col: str) -> nx.DiGraph:
        """
        Create a networkx representation of the significant associations
        :param essentiality_col: Which column to use for the essentiality property of the nodes
        :param essentiality_df: Dataframe with essentiality information, indexed by gene
        :return:graph
            A networkx DiGraph
        """
        network = nx.DiGraph()
        for kinase in self.kinase_dict.keys():
            network.add_node(kinase, type="Kinase", essential="NA")
        for tf in self.tf_dict.keys():
            network.add_node(tf, type="TF", essential="NA")
        for gene in self.targeted_genes:
            network.add_node(gene, type="Target_Gene", essential=essentiality_df[essentiality_col])
        associations = self.create_associations_information_df()
        for index, series in associations.iterrows():
            gene = series["gene"]
            tf = series["TF"]
            kinase = series["kinase"]
            network.add_edge(kinase, tf, influence=0)
            network.add_edge(tf, gene, influence=np.sign(associations.loc[index, "tf_coef"]))
        return network

    def create_cytoscape_from_associations(self,
                                           essentiality_df: pd.DataFrame,
                                           essentiality_col: str,
                                           outfile: str):
        """
        Save the associations as a cytoscape json file (.cyjs)
        :param essentiality_df: Passed to create_network_from_associations method
        :param essentiality_col: Passed to create_network_from_associations method
        :param outfile: File to write the cytoscape data to
        :return:
        """
        network = self.create_network_from_associations(essentiality_df=essentiality_df,
                                                        essentiality_col=essentiality_col)
        cyto_network = nx.cytoscape_data(network, name="name", ident="id")
        with open(outfile, "w") as f:
            json.dump(cyto_network, f)

    @staticmethod
    def compute_probabilities(data: pd.Series, direction: str = "over") -> pd.Series:
        """
        Function to compute probabilities from regression output
        :param data: Data to convert
        :param direction: Which direction the integral should be computed for the CDF,
            Starting from -inf: [over, left, l] (case insensitive)
            Going to -inf: [under, right, r] (case insensitive)
        :return: probabilities
            Pandas series of the probabilities, same shape as data
        """
        dist = gaussian_kde(data)
        probabilities = pd.Series(0, index=data.index)
        for sample in data.index:
            if direction.upper() in ["OVER", "LEFT", "L"]:
                probabilities[sample] = dist.integrate_box_1d(-np.inf, data[sample])
            elif direction.upper() in ["UNDER", "RIGHT", "R", "DOWN", "KO"]:
                probabilities[sample] = dist.integrate_box_1d(data[sample], np.inf)
            else:
                raise ValueError("Couldn't interpret direction")
        return probabilities
