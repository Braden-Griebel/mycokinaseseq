# Imports
import json
import typing
import warnings
from typing import Union, IO

# External library imports
import networkx as nx
import numpy as np
import pandas as pd
import statsmodels.api as sm
from scipy.stats import gaussian_kde
from sklearn.metrics import mean_squared_error, \
    mean_absolute_error, r2_score, median_absolute_error, \
    roc_auc_score, balanced_accuracy_score, f1_score, precision_score, recall_score, accuracy_score

# Local Imports
import mycokinaseseq.list_helper as lh
from mycokinaseseq import progress_bar

# Stop warnings about chained assignment
pd.options.mode.chained_assignment = None

# Global Variable setup
# Create a dictionary for converting Pkn.. to Rv..
KINASE_LOCUS_DICT = {
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
LOCUS_KINASE_DICT = {v: k for k, v in KINASE_LOCUS_DICT.items()}


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


def df_reduce(df1, df2, func: typing.Callable, axis: int = 1, **kwargs):
    """
    Function to take two dataframes, and reduce them
    :param df1: Pandas dataframe to reduce with df2 using the provided function
    :param df2: Pandas dataframe, must have same columns and index as df1
    :param func: Function which takes in two vectors (pandas Series) and returns a float
    :param axis: Axis to reduce over, 0 for rows, 1 for columns
    :return: reduced
        pandas series containing the value for each row or column (depending on axis argument)
    """
    # Make sure the index and columns are equal
    if not df1.index.equals(df2):
        raise ValueError("Indices are not equal")
    if not df1.columns.equals(df2):
        raise ValueError("Columns are not equal")
    # If the axis is 0, then iterate through the rows, combining each using the provided function
    if axis == 0:
        reduced = pd.Series(None, index=df1.index, dtype="Float64")
        for row in df1.index:
            # TODO: Add Try Except to catch issues if no examples of one label are present
            reduced[row] = func(df1.loc[row], df2.loc[row])
    # If the axis is 1, then iterate through the columns, combining each using the provided function
    elif axis == 1:
        reduced = pd.Series(None, index=df1.columns, dtype="Float64")
        for col in df1.columns:
            # TODO: Add Try Except to catch issues if no examples of one label are present
            reduced[col] = func(df1[col], df2[col], **kwargs)
    else:
        reduced = None
    return reduced


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

    def _setup_score_functions(self, kinase_expression: pd.DataFrame,
                               tf_expression: pd.DataFrame,
                               gene_expression: pd.DataFrame) -> (pd.DataFrame, pd.DataFrame):
        """
        Function to set up for the various score functions
        :param kinase_expression: Kinase expression values
        :param tf_expression: Transcription factor expression values
        :param gene_expression: Gene expression values
        :return: (gene_expression, predicted_expression)
            Tuple of pandas dataframes, the first representing the gene expression, the second representing the
            predicted expression
        """
        # First, check that all expression dataframes have the same samples
        samples = gene_expression.index
        # Replace all NA values with zeroes
        tf_expression = tf_expression.fillna(0)
        kinase_expression = kinase_expression.fillna(0)
        if not kinase_expression.index.equals(samples):
            raise ValueError("Kinase expression has different samples than gene expression")
        if not tf_expression.index.equals(samples):
            raise ValueError("TF expression has different samples than gene expression")
        # Compute the predicted values
        predicted = self.predict(tf_expression=tf_expression, kinase_expression=kinase_expression)
        # Check if gene_expression contains all predicted genes, if it does then take the subset of the predicted
        #   dataframe to ensure that the predicted dataframe has the same columns as the gene expression dataframe
        if not set(predicted.columns).issubset(set(gene_expression.columns)):
            raise ValueError("Gene expression dataframe doesn't include all predicted genes")
        gene_expression = gene_expression[predicted.columns]
        return gene_expression, predicted

    def score_gene_regression(self,
                              kinase_expression: pd.DataFrame,
                              tf_expression: pd.DataFrame,
                              gene_expression: pd.DataFrame,
                              metric: typing.Callable,
                              **kwargs) -> pd.Series:
        """
        Function to score the regression model using the provided metric, returns a score for each gene
        :param kinase_expression: Expression values of the kinases, dimensions are (n_samples, n_kinases)
        :param tf_expression: Expression values of the transcription factors, dimensions are (n_samples, n_tfs)
        :param gene_expression: Expression values of the targeted genes, dimensions are (n_samples, genes)
        :param metric: Metric to use for scoring, should take two pandas series and return a Float64
        :param kwargs: Key word arguments passed to metric
        :return: regression_gene_scores
            Pandas series indexed by gene, with values representing the score for each gene model
        """
        # Call setup to check index and columns, and make sure the gene_expression and prediction have the same columns
        gene_expression, predicted = self._setup_score_functions(kinase_expression=kinase_expression,
                                                                 tf_expression=tf_expression,
                                                                 gene_expression=gene_expression)
        regression_gene_scores = df_reduce(gene_expression, predicted, metric, axis=1, **kwargs)
        return regression_gene_scores

    def score_gene_classification_labels(self,
                                         kinase_expression: pd.DataFrame,
                                         tf_expression: pd.DataFrame,
                                         gene_expression: pd.DataFrame,
                                         cutoff: float = 1.,
                                         cutoff_method: str = "simple",
                                         metric: typing.Callable = f1_score,
                                         **kwargs: dict) -> (pd.Series, pd.Series):
        """
        Function to score the model gene-by-gene based on classification
        :param kinase_expression: Expression values of the kinases, dimensions are (n_samples, n_kinases)
        :param tf_expression: Expression values of the transcription factors, dimensions are (n_samples, n_tfs)
        :param gene_expression: Expression values of the targeted genes, dimensions are (n_samples, genes)
        :param cutoff: Value for cutoff, either an expression level, or probability depending on cutoff method
        :param cutoff_method: How to perform binarization, either
            "simple" for considering over expression to be any value greater than cutoff and under expression to be
                any value below negative cutoff
            "probability" for considering over expression to be any value in the top cutoff proportion of the data,
                and under expression to be any value in the bottom cutoff proportion of the data
        :param metric: Metric to compute, should take two pandas series, and return a score (true is first parameter,
            predicted is the second parameter). Both will correspond to labels.
        :return: oe_scores, ue_scores
            Pandas dataframes representing the over expression, and under expression scores
        """
        # Setup
        gene_expression, predicted = self._setup_score_functions(kinase_expression=kinase_expression,
                                                                 tf_expression=tf_expression,
                                                                 gene_expression=gene_expression)
        # Perform binarization, for both gene_expression and predicted dataframes
        if cutoff_method.upper() in ["SIMPLE", "S", "NORMAL"]:
            oe_true_labels = (gene_expression > cutoff)
            ue_true_labels = (gene_expression < -1 * cutoff)
            oe_pred_labels = (predicted > cutoff)
            ue_pred_labels = (predicted < -1 * cutoff)
        elif cutoff_method.upper() in ["PROB", "PROP", "PROBABILITY", "PROPORTION", "S"]:
            probability_true = pd.DataFrame(None, index=gene_expression.index, columns=gene_expression.columns)
            probability_pred = pd.DataFrame(None, index=predicted.index, columns=predicted.columns)
            for gene in gene_expression.columns:
                probability_true = compute_probabilities(gene_expression[gene], direction="left")
                probability_pred = compute_probabilities(predicted[gene], direction="left")
            oe_true_labels = (probability_true > 1 - cutoff)
            ue_true_labels = (probability_true < cutoff)
            oe_pred_labels = (probability_pred > 1 - cutoff)
            ue_pred_labels = (probability_pred < cutoff)
        else:
            raise ValueError("Couldn't interpret cutoff method")
        oe_scores = df_reduce(oe_true_labels, oe_pred_labels, func=metric, axis=1, **kwargs)
        ue_scores = df_reduce(ue_true_labels, ue_pred_labels, func=metric, axis=1, **kwargs)
        return oe_scores, ue_scores

    def score_gene_classification_decision_function(self,
                                                    kinase_expression: pd.DataFrame,
                                                    tf_expression: pd.DataFrame,
                                                    gene_expression: pd.DataFrame,
                                                    cutoff: float = 1.,
                                                    cutoff_method: str = "simple",
                                                    metric: typing.Callable = roc_auc_score,
                                                    **kwargs):
        """
        Function to score the model gene-by-gene based on classification
        :param kinase_expression: Expression values of the kinases, dimensions are (n_samples, n_kinases)
        :param tf_expression: Expression values of the transcription factors, dimensions are (n_samples, n_tfs)
        :param gene_expression: Expression values of the targeted genes, dimensions are (n_samples, genes)
        :param cutoff: Value for cutoff, either an expression level, or probability depending on cutoff method
        :param cutoff_method: How to perform binarization, either
            "simple" for considering over expression to be any value greater than cutoff and under expression to be
                any value below negative cutoff
            "probability" for considering over expression to be any value in the top cutoff proportion of the data,
                and under expression to be any value in the bottom cutoff proportion of the data
        :param metric: Metric to compute, should take two pandas series, and return a score (true is first parameter,
            predicted is the second parameter). True will be boolean labels, while predicted will be the values
            of the decision function.
        :param kwargs: keyword arguments passed to metric
        :return: oe_scores, ue_scores
            Pandas dataframes representing the over expression, and under expression scores
        """
        # Setup
        gene_expression, predicted = self._setup_score_functions(kinase_expression=kinase_expression,
                                                                 tf_expression=tf_expression,
                                                                 gene_expression=gene_expression)
        # Perform binarization for gene_expression
        if cutoff_method.upper() in ["SIMPLE", "S", "NORMAL"]:
            oe_true_labels = (gene_expression > cutoff)
            ue_true_labels = (gene_expression < -1 * cutoff)
        elif cutoff_method.upper() in ["PROB", "PROP", "PROBABILITY", "PROPORTION", "S"]:
            probability_true = pd.DataFrame(None, index=gene_expression.index, columns=gene_expression.columns)
            for gene in gene_expression.columns:
                probability_true = compute_probabilities(gene_expression[gene], direction="left")
            oe_true_labels = (probability_true > 1 - cutoff)
            ue_true_labels = (probability_true < cutoff)
        else:
            raise ValueError("Couldn't Interpret Cutoff Method")
        oe_scores = df_reduce(oe_true_labels, predicted, func=metric, axis=1, **kwargs)
        ue_scores = df_reduce(ue_true_labels, -1 * predicted, func=metric, axis=1, **kwargs)
        return oe_scores, ue_scores

    def score_overall_regression(self,
                                 kinase_expression: pd.DataFrame,
                                 tf_expression: pd.DataFrame,
                                 gene_expression: pd.DataFrame,
                                 metric: typing.Callable,
                                 **kwargs) -> float:
        """
        Function to score the regression model using the provided metric, returns an overall score
        :param kinase_expression: Expression values of the kinases, dimensions are (n_samples, n_kinases)
        :param tf_expression: Expression values of the transcription factors, dimensions are (n_samples, n_tfs)
        :param gene_expression: Expression values of the targeted genes, dimensions are (n_samples, genes)
        :param metric: Metric to use for scoring, should take two 1d numpy arrays and return a Float64
        :param kwargs: key word arguments passed to metric
        :return: score
            Value for the score across the entire model
        """
        # Setup
        gene_expression, predicted = self._setup_score_functions(kinase_expression=kinase_expression,
                                                                 tf_expression=tf_expression,
                                                                 gene_expression=gene_expression)
        # Make sure the predicted dataframe matches the gene expression data frame
        predicted = predicted.loc[gene_expression.index, gene_expression.columns]
        # Flatten the gene_expression, and the predicted dataframes
        flat_gene_expression = gene_expression.to_numpy().reshape((-1, 1))
        flat_pred_expression = predicted.to_numpy().reshape((-1, 1))
        # Score the model and return the result
        return metric(flat_gene_expression, flat_pred_expression, **kwargs)

    def score_overall_classification_labels(self,
                                            kinase_expression: pd.DataFrame,
                                            tf_expression: pd.DataFrame,
                                            gene_expression: pd.DataFrame,
                                            cutoff: float = 1.,
                                            cutoff_method: str = "simple",
                                            metric: typing.Callable = f1_score,
                                            **kwargs):
        """
        Function to score the model overall based on classification
        :param kinase_expression: Expression values of the kinases, dimensions are (n_samples, n_kinases)
        :param tf_expression: Expression values of the transcription factors, dimensions are (n_samples, n_tfs)
        :param gene_expression: Expression values of the targeted genes, dimensions are (n_samples, genes)
        :param cutoff: Value for cutoff, either an expression level, or probability depending on cutoff method
        :param cutoff_method: How to perform binarization, either
            "simple" for considering over expression to be any value greater than cutoff and under expression to be
                any value below negative cutoff
            "probability" for considering over expression to be any value in the top cutoff proportion of the data,
                and under expression to be any value in the bottom cutoff proportion of the data
        :param metric: Metric to compute, should take two pandas series, and return a score (true is first parameter,
            predicted is the second parameter). Both will correspond to labels.
        :param kwargs: Keyword arguments passed to metric function
        :return: oe_score, ue_score
            tuple of scores for over expression and under expression
        """
        # Setup
        gene_expression, predicted = self._setup_score_functions(kinase_expression=kinase_expression,
                                                                 tf_expression=tf_expression,
                                                                 gene_expression=gene_expression)
        # Ensure that the predicted dataframe has the same index and columns as the gene_expression, in the same order
        predicted = predicted.loc[gene_expression.index, gene_expression.columns]
        # Perform binarization, for both gene_expression and predicted dataframes
        if cutoff_method.upper() in ["SIMPLE", "S", "NORMAL"]:
            oe_true_labels = (gene_expression > cutoff)
            ue_true_labels = (gene_expression < -1 * cutoff)
            oe_pred_labels = (predicted > cutoff)
            ue_pred_labels = (predicted < -1 * cutoff)
        elif cutoff_method.upper() in ["PROB", "PROP", "PROBABILITY", "PROPORTION", "S"]:
            probability_true = pd.DataFrame(None, index=gene_expression.index, columns=gene_expression.columns)
            probability_pred = pd.DataFrame(None, index=predicted.index, columns=predicted.columns)
            for gene in gene_expression.columns:
                probability_true = compute_probabilities(gene_expression[gene], direction="left")
                probability_pred = compute_probabilities(predicted[gene], direction="left")
            oe_true_labels = (probability_true > 1 - cutoff)
            ue_true_labels = (probability_true < cutoff)
            oe_pred_labels = (probability_pred > 1 - cutoff)
            ue_pred_labels = (probability_pred < cutoff)
        else:
            raise ValueError("Couldn't Interpret Cutoff Method")
        # Flatten the arrays
        flat_oe_true_labels = oe_true_labels.to_numpy().reshape((-1, 1))
        flat_ue_true_labels = ue_true_labels.to_numpy().reshape((-1, 1))
        flat_oe_pred_labels = oe_pred_labels.to_numpy().reshape((-1, 1))
        flat_ue_pred_labels = ue_pred_labels.to_numpy().reshape((-1, 1))
        # Score the model using the metric
        oe_score = metric(flat_oe_true_labels, flat_oe_pred_labels, **kwargs)
        ue_score = metric(flat_ue_true_labels, flat_ue_pred_labels, **kwargs)
        return oe_score, ue_score

    def score_overall_classification_decision_function(self,
                                                       kinase_expression: pd.DataFrame,
                                                       tf_expression: pd.DataFrame,
                                                       gene_expression: pd.DataFrame,
                                                       cutoff: float = 1.,
                                                       cutoff_method: str = "simple",
                                                       metric: typing.Callable = roc_auc_score,
                                                       **kwargs):
        """
        Function to score the model gene-by-gene based on classification
        :param kinase_expression: Expression values of the kinases, dimensions are (n_samples, n_kinases)
        :param tf_expression: Expression values of the transcription factors, dimensions are (n_samples, n_tfs)
        :param gene_expression: Expression values of the targeted genes, dimensions are (n_samples, genes)
        :param cutoff: Value for cutoff, either an expression level, or probability depending on cutoff method
        :param cutoff_method: How to perform binarization, either
            "simple" for considering over expression to be any value greater than cutoff and under expression to be
                any value below negative cutoff
            "probability" for considering over expression to be any value in the top cutoff proportion of the data,
                and under expression to be any value in the bottom cutoff proportion of the data
        :param metric: Metric to compute, should take two pandas series, and return a score (true is first parameter,
            predicted is the second parameter). True will be boolean labels, while predicted will be the values
            of the decision function.
        :param kwargs: keyword arguments passed to metric
        :return: oe_scores, ue_scores
            Pandas dataframes representing the over expression, and under expression scores
        """
        # Setup
        gene_expression, predicted = self._setup_score_functions(kinase_expression=kinase_expression,
                                                                 tf_expression=tf_expression,
                                                                 gene_expression=gene_expression)
        # Ensure that the predicted dataframe has its index and columns match the gene_expression dataframe
        predicted = predicted.loc[gene_expression.index, gene_expression.columns]
        # Perform binarization for gene_expression
        if cutoff_method.upper() in ["SIMPLE", "S", "NORMAL"]:
            oe_true_labels = (gene_expression > cutoff)
            ue_true_labels = (gene_expression < -1 * cutoff)
        elif cutoff_method.upper() in ["PROB", "PROP", "PROBABILITY", "PROPORTION", "S"]:
            probability_true = pd.DataFrame(None, index=gene_expression.index, columns=gene_expression.columns)
            for gene in gene_expression.columns:
                probability_true = compute_probabilities(gene_expression[gene], direction="left")
            oe_true_labels = (probability_true > 1 - cutoff)
            ue_true_labels = (probability_true < cutoff)
        else:
            raise ValueError("Couldn't Interpret Cutoff method")
        # Flatten the dataframes
        flat_oe_true_labels = oe_true_labels.to_numpy().reshape((-1, 1))
        flat_ue_true_labels = ue_true_labels.to_numpy().reshape((-1, 1))
        flat_predicted = predicted.to_numpy().reshape((-1, 1))
        # Score the model
        oe_score = metric(flat_oe_true_labels, flat_predicted, **kwargs)
        ue_score = metric(flat_ue_true_labels, -1 * flat_predicted, **kwargs)
        return oe_score, ue_score

    def score_restricted(self,
                         compendia: pd.DataFrame,
                         intercept: bool = True,
                         verbose: bool = False,
                         use_significant_associations: bool = False,
                         association_kwargs: dict = None,
                         **kwargs
                         ):
        """
        Function to use restriction F-tests to evaluate the model in comparison to models using only TFs, and
            models using only kinases
        :param association_kwargs: keyword arguments to pass to the find_associations method
        :param use_significant_associations: Whether the significant associations should be used for the full model,
            or the full model should include all possible interaction terms regardless of their individual significance
        :param compendia: RNA seq gene expression compendia, in log2(fold-change) form, dimensions are
            (n_samples, n_targeted_genes+n_kinases+n_tfs),
        :param intercept: Whether to include an intercept in the model
        :param verbose: Whether a verbose output is desired
        :param kwargs: Key word arguments passed to the statsmodels regression model fit method
        :return: prob_df
            Pandas dataframe of probabilities from the restricted F-test
        """
        if verbose:
            print("Evaluating restricted models")
            bar = progress_bar.ProgressBar(total=len(self.targeted_genes), divisions=10)
        else:
            bar = None
        prob_df = pd.DataFrame(0,
                               index=self.targeted_genes,
                               columns=["tf_only", "kinase_only", "tf_kinase"])
        # If the significant associations are the ones that should be used for
        if use_significant_associations:
            if verbose:
                print("Finding significant associations")
            if not self.significant_associations:
                if not association_kwargs:
                    association_kwargs = {
                        "significance_level": 0.05,
                        "multi_comparison_method": "bh",
                        "false_discovery_rate": 0.05,
                        "intercept": True,
                        "verbose": verbose
                    }
                self.find_associations(compendia, **association_kwargs)
        # Iterate through the genes, and compute the needed probabilities
        for gene in self.targeted_genes:
            if verbose:
                # noinspection PyUnboundLocalVariables
                bar.inc()
            data = compendia.copy()
            if use_significant_associations:
                interaction_terms = self.find_interaction_terms_from_associations(gene, self.significant_associations)
            else:
                interaction_terms = self.find_interaction_terms_initial(gene)
            # Create the full model
            exogenous_array = self.create_exogenous_array(compendia=compendia,
                                                          gene=gene,
                                                          interaction_terms=interaction_terms,
                                                          intercept=intercept).dropna(axis=0)
            endogenous_array = data[gene].dropna(axis=0)
            defined_samples = lh.find_intersect(list(exogenous_array.index), list(endogenous_array.index))
            exogenous_array = exogenous_array.loc[defined_samples]
            endogenous_array = endogenous_array.loc[defined_samples]
            full_model = sm.OLS(endog=endogenous_array, exog=exogenous_array)
            full_results = full_model.fit()

            # Create the tf only model arrays
            tf_exog = data[self.gene_to_tf_dict[gene]].loc[defined_samples]
            # Create the tf_only model
            tf_model = sm.OLS(endog=endogenous_array, exog=tf_exog)
            tf_results = tf_model.fit(**kwargs)

            # Create the kinase only model arrays
            kinase_exog = data[self.gene_to_tf_to_kinase_dict[gene]].loc[defined_samples]
            kinase_model = sm.OLS(endog=endogenous_array, exog=kinase_exog)
            kinase_results = kinase_model.fit(**kwargs)

            # Create the TF, Kinase Model (not including interaction terms)
            tf_kinase_exog = data[self.gene_to_tf_dict[gene] + self.gene_to_tf_to_kinase_dict[gene]].loc[
                defined_samples]
            tf_kinase_model = sm.OLS(endog=endogenous_array, exog=tf_kinase_exog)
            tf_kinase_results = tf_kinase_model.fit(**kwargs)

            # Add the results of the compare_f_test to the prob_df
            _, prob_df.loc[gene, "tf_only"], _ = full_results.compare_f_test(tf_results)
            _, prob_df.loc[gene, "kinase_only"], _ = full_results.compare_f_test(kinase_results)
            _, prob_df.loc[gene, "tf_kinase"], _ = full_results.compare_f_test(tf_kinase_results)

        # Return the completed prob_df
        return prob_df

    def score_full(self,
                   kinase_expression: pd.DataFrame,
                   tf_expression: pd.DataFrame,
                   gene_expression: pd.DataFrame,
                   cutoff: float = 1.,
                   cutoff_method: str = "simple",
                   regression_metric_list: list = None,
                   regression_metric_names: list = None,
                   regression_metric_kwargs: list = None,
                   classification_labels_metric_list: list = None,
                   classification_labels_metric_names: list = None,
                   classification_labels_metric_kwargs: list = None,
                   classification_decision_function_metric_list: list = None,
                   classification_decision_function_metric_names: list = None,
                   classification_decision_function_metric_kwargs: list = None,
                   ) -> (pd.DataFrame, pd.Series):
        """
        Function to score the model gene-by-gene based on classification
        :param kinase_expression: Expression values of the kinases, dimensions are (n_samples, n_kinases)
        :param tf_expression: Expression values of the transcription factors, dimensions are (n_samples, n_tfs)
        :param gene_expression: Expression values of the targeted genes, dimensions are (n_samples, genes)
        :param cutoff: Value for cutoff, either an expression level, or probability depending on cutoff method
        :param cutoff_method: How to perform binarization, either
            "simple" for considering over expression to be any value greater than cutoff and under expression to be
                any value below negative cutoff
            "probability" for considering over expression to be any value in the top cutoff proportion of the data,
                and under expression to be any value in the bottom cutoff proportion of the data
        :param regression_metric_list list of metrics to use for regression
        :param regression_metric_names list of the names of the regression metrics
        :param regression_metric_kwargs list of kwarg dicts passes to the regression metrics
        :param classification_labels_metric_list list of metrics to use for scoring classification labels
        :param classification_labels_metric_names list of names for the metrics used for scoring classification labels
        :param classification_labels_metric_kwargs list of kwarg dicts for the metrics used for scoring classification
            labels
        :param classification_decision_function_metric_list List of the metrics to use for scoring the classification
            decision functions
        :param classification_decision_function_metric_names List of names for the metrics used for scoring the
            classification decision functions
        :param classification_decision_function_metric_kwargs List of kwargs dicts for the metrics used for scoring the
            classification decision functions
        :return: gene_scores, overall_scores
            Tuple of gene-by-gene scores in dataframe, and overall scores in a pandas series
        """
        # Evaluation of function arguments
        # If there is not a provided regression metric list, create one, along with the names, and kwarg dicts
        if not regression_metric_list:
            regression_metric_list = [mean_squared_error,
                                      mean_absolute_error,
                                      median_absolute_error,
                                      r2_score]
            regression_metric_names = ["mean_squared_error",
                                       "mean_absolute_error",
                                       "median_absolute_error",
                                       "r2"]
            regression_metric_kwargs = [dict(), dict(), dict(), dict()]
        # If there is provided regression metric lists, make sure it is the same length as the metric names
        #   and ensure that if it is not none, the regression metric kwargs list is the same length as well
        else:
            if not regression_metric_names:
                regression_metric_names = []
                for i in range(len(regression_metric_list)):
                    regression_metric_names.append(f"r_{i}")
            else:
                if len(regression_metric_names) != len(regression_metric_list):
                    raise ValueError("Regression Metric List and Regression Metric Names must be the same length")
            if not regression_metric_kwargs:
                regression_metric_kwargs = []
                for _ in range(len(regression_metric_list)):
                    regression_metric_kwargs.append(dict())
            else:
                if len(regression_metric_kwargs) != len(regression_metric_list):
                    raise ValueError("Regression Metric List and Regression Metric Kwargs must be the same length")
        # Similarly for the classification label scores
        # If classification label metrics are not provided, create list, names, and kwargs
        if not classification_labels_metric_list:
            classification_labels_metric_list = [
                accuracy_score,
                precision_score,
                recall_score,
                f1_score,
                balanced_accuracy_score,
            ]
            classification_labels_metric_names = [
                "accuracy",
                "precision",
                "recall",
                "f1",
                "balanced_accuracy"
            ]
            classification_labels_metric_kwargs = [
                dict(),
                dict(),
                dict(),
                dict(),
                dict()
            ]
        else:
            # If there is not a list of names for the metrics, create one
            if not classification_labels_metric_names:
                classification_labels_metric_names = []
                for i in range(len(classification_labels_metric_list)):
                    classification_labels_metric_names.append(f"cl_{i}")
            # If there is a list of names, make sure it is the same length as the metric list
            else:
                if len(classification_labels_metric_names) != len(classification_labels_metric_list):
                    raise ValueError("Classification Label Metric List and Classification Label Metric names must be "
                                     "the same length")
            # If there is not a list of kwarg dicts, create one
            if not classification_labels_metric_kwargs:
                classification_labels_metric_kwargs = []
                for _ in range(len(classification_labels_metric_list)):
                    classification_labels_metric_kwargs.append(dict())
            # If there is, make sure it is the same length as the list of metrics
            else:
                if len(classification_labels_metric_kwargs) != len(classification_labels_metric_list):
                    raise ValueError("Classification Label Metric List and Classification Label Metric Kwargs must be "
                                     "the same length")
        # And finally for the Classification decision function
        # If there is not a classification decision function metric list
        if not classification_decision_function_metric_list:
            classification_decision_function_metric_list = [roc_auc_score]
            classification_decision_function_metric_names = ["roc_auc_score"]
            classification_decision_function_metric_kwargs = [dict()]
        # If there is, check the name and kwargs arguments
        else:
            # If there is not a list of names, create one
            if not classification_decision_function_metric_names:
                classification_decision_function_metric_names = []
                for i in range(len(classification_decision_function_metric_list)):
                    classification_decision_function_metric_names.append(f"cf_{i}")
            # If there is a list of names, make sure it is the same length as the metric list
            else:
                if (len(classification_decision_function_metric_names) !=
                        len(classification_decision_function_metric_list)):
                    raise ValueError("Classification decision function names must be the same length as the provided "
                                     "metric list")
            # If there is not a list of kwarg dicts, create one
            if not classification_decision_function_metric_kwargs:
                classification_decision_function_metric_kwargs = []
                for _ in range(len(classification_decision_function_metric_list)):
                    classification_decision_function_metric_kwargs.append(dict())
            # If there is, check that it is the right length
            else:
                if (len(classification_decision_function_metric_list) !=
                        len(classification_decision_function_metric_kwargs)):
                    raise ValueError("Classification decision function kwarg list must be the same length as the "
                                     "provided metric list")
        # Now that all the metric, names, and kwarg lists have been either created or checked,
        gene_score_list = []
        overall_score_dict = dict()
        for metric, kwarg_dict, name in zip(regression_metric_list, regression_metric_kwargs, regression_metric_names):
            gene_regression_scores = self.score_gene_regression(kinase_expression=kinase_expression,
                                                                tf_expression=tf_expression,
                                                                gene_expression=gene_expression,
                                                                metric=metric, **kwarg_dict)
            gene_regression_scores.name = f"regression_{name}"
            gene_score_list.append(gene_regression_scores)
            overall_regression_scores = self.score_overall_regression(kinase_expression=kinase_expression,
                                                                      tf_expression=tf_expression,
                                                                      gene_expression=gene_expression,
                                                                      metric=metric, **kwarg_dict)
            overall_score_dict[f"regression_{name}"] = overall_regression_scores
        for metric, kwarg_dict, name in zip(classification_labels_metric_list,
                                            classification_labels_metric_kwargs,
                                            classification_labels_metric_names):
            gene_oe_cl_scores, gene_ue_cl_scores = \
                self.score_gene_classification_labels(kinase_expression=kinase_expression,
                                                      tf_expression=tf_expression,
                                                      gene_expression=gene_expression,
                                                      cutoff=cutoff,
                                                      cutoff_method=cutoff_method,
                                                      metric=metric,
                                                      **kwarg_dict)
            gene_oe_cl_scores.name = f"cl_{name}_oe"
            gene_ue_cl_scores.name = f"cl_{name}_ue"
            gene_score_list.append(gene_oe_cl_scores)
            gene_score_list.append(gene_ue_cl_scores)
            overall_cl_oe_score, overall_cl_ue_score = \
                self.score_overall_classification_labels(kinase_expression=kinase_expression,
                                                         tf_expression=tf_expression,
                                                         gene_expression=gene_expression,
                                                         cutoff=cutoff,
                                                         cutoff_method=cutoff_method,
                                                         metric=metric,
                                                         **kwarg_dict)
            overall_score_dict[f"cl_{name}_oe"] = overall_cl_oe_score
            overall_score_dict[f"cl_{name}_ue"] = overall_cl_ue_score
        for metric, kwarg_dict, name in zip(classification_decision_function_metric_list,
                                            classification_decision_function_metric_kwargs,
                                            classification_decision_function_metric_names):
            gene_oe_cf_scores, gene_ue_cf_scores = \
                self.score_gene_classification_decision_function(kinase_expression=kinase_expression,
                                                                 tf_expression=tf_expression,
                                                                 gene_expression=gene_expression,
                                                                 cutoff=cutoff,
                                                                 cutoff_method=cutoff_method,
                                                                 metric=metric,
                                                                 **kwarg_dict)
            gene_oe_cf_scores.name = f"cf_{name}_oe"
            gene_ue_cf_scores.name = f"cf_{name}_ue"
            gene_score_list.append(gene_oe_cf_scores)
            gene_score_list.append(gene_ue_cf_scores)
            overall_cf_oe_score, overall_cf_ue_score = \
                self.score_overall_classification_decision_function(kinase_expression=kinase_expression,
                                                                    tf_expression=tf_expression,
                                                                    gene_expression=gene_expression,
                                                                    cutoff=cutoff,
                                                                    cutoff_method=cutoff_method,
                                                                    metric=metric,
                                                                    **kwarg_dict)
            overall_score_dict[f"cf_{name}_oe"] = overall_cf_oe_score
            overall_score_dict[f"cf_{name}_ue"] = overall_cf_ue_score
        # Combine the gene scores into a dataframe, and create a series from the overall scores, then return these
        gene_scores = pd.concat(gene_score_list, axis=1)
        overall_scores = pd.Series(overall_score_dict)
        return gene_scores, overall_scores

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
                               interaction_terms: list[str],
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
        term_list = tfs + kinases + interaction_terms
        if intercept:
            term_list.append("intercept")
        exogenous_array = compendia[term_list]
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
                          intercept: bool = True,
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
        # Compute the full compendia, with all possible interaction terms to avoid recalculating it repeatedly
        compendia_full = compendia.copy()
        interaction_terms_series_list = []
        intercept_series = pd.Series(1., index=compendia_full.index)
        intercept_series.name = "intercept"
        for tf in self.tf_list:
            if tf in self.tf_kinase_dict:
                for kinase in self.tf_kinase_dict[tf]:
                    interaction_terms_series_list.append((compendia[kinase] * compendia[tf]))
                    interaction_terms_series_list[-1].name = f"{kinase}_{tf}"
        compendia_full = pd.concat([compendia_full] + interaction_terms_series_list + [intercept_series], axis=1)
        associations_dict = {
            "kinase": [],
            "kinase_name": [],
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
            interaction_terms = self.interaction_term_repr_list(self.find_interaction_terms_initial(gene))
            exogenous_array = self.create_exogenous_array(compendia=compendia_full,
                                                          gene=gene,
                                                          interaction_terms=interaction_terms,
                                                          intercept=intercept).dropna(axis=0)
            endogenous_array = compendia_full[gene].dropna(axis=0)
            defined_samples = lh.find_intersect(list(exogenous_array.index), list(endogenous_array.index))
            exogenous_array = exogenous_array.loc[defined_samples]
            endogenous_array = endogenous_array.loc[defined_samples]
            model = sm.OLS(endog=endogenous_array, exog=exogenous_array)
            results = model.fit()
            for representation in interaction_terms:
                kinase, tf = representation.split("_")
                associations_dict["kinase"].append(kinase)
                associations_dict["TF"].append(tf),
                associations_dict["gene"].append(gene),
                associations_dict["p_value"].append(results.pvalues[representation]),
                associations_dict["r_squared"].append(results.rsquared),
                associations_dict["adjusted_r_squared"].append(results.rsquared_adj),
                associations_dict["tf_coef"].append(results.params[tf]),
                associations_dict["kinase_coef"].append(results.params[kinase]),
                associations_dict["interaction_coef"].append(results.params[representation])
                associations_dict["kinase_name"].append(LOCUS_KINASE_DICT[kinase])
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
        # Check for L1_wt, and refit, since those will lead to errors
        if "refit" in kwargs.keys():
            if kwargs["refit"]:
                raise ValueError("refit can't be used due to issues with statsmodels return type")
        if "L1_wt" in kwargs.keys():
            if kwargs["L1_wt"] == 0 or kwargs["L1_wt"] == 0.:
                raise ValueError("L1_wt can't be zero due to issues with statsmodels return type")
        # If a verbose output is desired, create the progress bar object
        if verbose:
            print("Finding Model Coefficients")
            bar = progress_bar.ProgressBar(total=len(self.targeted_genes), divisions=10)
        else:
            bar = None
        # Calculate the full compendia containing all possible interaction terms, and keep a list of interaction terms
        all_interaction_terms_list = []
        compendia_full = compendia.copy()
        interaction_term_series_list = []
        intercept_series = pd.Series(1., index=compendia_full.index)
        intercept_series.name = "intercept"
        for tf in self.tf_dict.keys():
            if tf in self.tf_kinase_dict:
                for kinase in self.tf_kinase_dict[tf]:
                    all_interaction_terms_list.append(f"{kinase}_{tf}")
                    interaction_term_series_list.append((compendia_full[kinase] * compendia_full[tf]))
                    interaction_term_series_list[-1].name = f"{kinase}_{tf}"
        compendia_full = pd.concat([compendia_full] + interaction_term_series_list + [intercept_series], axis=1)
        # Create a dataframe to hold all the model coefficients
        self.model_coefficients = pd.DataFrame(data=0.,
                                               index=self.targeted_genes,
                                               columns=(list(self.tf_dict.keys()) +
                                                        list(self.kinase_dict.keys()) +
                                                        all_interaction_terms_list))
        # If intercept terms are desired, add a column for them to the full compendia and the model_coefficients
        if intercept:
            self.model_coefficients["intercept"] = 0.0
        # Iterate through all the targeted genes and create a model for each
        for gene in self.targeted_genes:
            # If verbose, print the progress
            if verbose:
                bar.inc()
            interaction_terms = self.interaction_term_repr_list(
                self.find_interaction_terms_from_associations(gene, self.significant_associations,
                                                              gene_col="gene",
                                                              tf_col="TF",
                                                              kinase_col="kinase"))
            # TODO: Change how the model deals with NA entries
            # Create the exogenous and endogenous arrays
            exogenous_array = self.create_exogenous_array(compendia=compendia_full,
                                                          gene=gene,
                                                          interaction_terms=interaction_terms,
                                                          intercept=intercept).dropna(axis=0)
            endogenous_array = compendia_full[gene].dropna(axis=0)
            # Find the samples which are fully defined (no NaN, so shared between the exogenous and endogenous arrays)
            defined_samples = lh.find_intersect(list(exogenous_array.index), list(endogenous_array.index))
            exogenous_array = exogenous_array.loc[defined_samples]
            endogenous_array = endogenous_array.loc[defined_samples]
            # Create the statsmodels object
            model = sm.OLS(endog=endogenous_array, exog=exogenous_array)
            # Fit the model, depending on if regularization is desired or not
            if regularized:
                results = model.fit_regularized(**kwargs)
            else:
                results = model.fit(**kwargs)
            # Get the coefficients from the results instance
            coefficients = results.params
            # Add all the coefficients to the model coefficients dataframe
            for coef in coefficients.index:
                self.model_coefficients.loc[gene, coef] = coefficients[coef]
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


class TfOnlyModel:
    def __init__(self, kinase_dict: dict, tf_dict: dict, gene_list: list):
        """
        init Function for TF Only Model
        :param kinase_dict: dict of kinase:gene showing the phosphorylation targets
        :param tf_dict: dict of tf:gene showing tf targets
        :param gene_list: Total list of genes which will be in the compendia
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
        # Variable for if the models have intercepts
        self.intercept = None
        # Variable for if the model is regularized
        self.regularized = None

    def fit(self, data: pd.DataFrame, regularized: bool, intercept: bool = True, verbose: bool = False, **kwargs):
        """
        Function to fit the model to data
        :param data: Data for fitting the model, dimensions are (n_samples, n_genes)
        :param regularized: Whether the model should be regularized
        :param intercept: Whether an intercept should be included in the model
        :param verbose: Whether a verbose output is desired
        :param kwargs: keyword arguments passed to stats models fit methods
        :return: None
        """
        # Setting up variable for later
        bar = None
        # Create the dataframe to hold the coefficients for the model
        self.model_coefficients = pd.DataFrame(0., index=self.targeted_genes, columns=self.tf_list)
        # If an intercept is desired for the model, set flag to true and add intercept column to the model coefficient
        #   dataframe
        if intercept:
            self.intercept = True
            self.model_coefficients["intercept"] = 1.
        # Check if a verbose output is desired, if it is set up progress bar
        if verbose:
            print("Finding Coefficients")
            bar = progress_bar.ProgressBar(total=len(self.targeted_genes), divisions=10)
        # Iterate through the genes, fit the models, record the coefficients
        for gene in self.targeted_genes:
            if verbose:
                bar.inc()
            # Create exogenous array
            exog = self.create_exogenous_array(gene=gene, data=data, intercept=intercept).dropna(axis=0)
            # Create the endogenous array
            endog = data[gene].dropna(axis=0)
            # since NA values are dropped, make sure the endog and exog arrays can still line up
            # TODO: May need to also remove values where the kinases targeting the gene are not here to match
            #   with the full model
            defined_samples = lh.find_intersect(list(exog.index), list(endog.index))
            exog = exog.loc[defined_samples]
            endog = endog.loc[defined_samples]
            # Create the ordinary least squares statsmodels
            model = sm.OLS(endog=endog, exog=exog)
            # Fit the model (depending on whether regularization is desired)
            self.regularized = regularized
            if self.regularized:
                results = model.fit_regularized(**kwargs)
            else:
                results = model.fit(**kwargs)
            # Get the coefficient values from the results instance
            coefficients = results.params
            # Iterate through the TFs targeting the gene, and add the coefficients for them to model_coefficients
            for tf in self.gene_to_tf_dict[gene]:
                self.model_coefficients.loc[gene, tf] = coefficients[tf]
            if intercept:
                self.model_coefficients.loc[gene, "intercept"] = coefficients["intercept"]
        if verbose:
            print("Found Coefficients")

    def predict(self, tf_expression: pd.DataFrame):
        """
        Function to predict gene expression based on TF expression
        :param tf_expression: Dataframe of tf expression, dimensions are (n_samples, n_tfs)
        :return: pred_gene_expression
            Dataframe with predicted gene expression, dimensions are (n_samples, n_genes)
        """
        input_df = tf_expression[self.model_coefficients.columns]
        output = pd.DataFrame(
            np.transpose(
                np.matmul(self.model_coefficients.to_numpy(),
                          np.transpose(input_df.fillna(0).to_numpy())
                          )
            ), index=tf_expression.index, columns=self.model_coefficients.index
        )
        return output

    def score(self, tf_expression: pd.DataFrame, gene_expression: pd.DataFrame, metric: typing.Callable):
        """
        Method to score the model against the provided gene_expression values
        :param tf_expression: TF expression dataframe, dimensions are (n_samples, n_tfs)
        :param gene_expression: Gene expression dataframe, dimensions are (n_samples, n_genes)
        :param metric: Metric to use for scoring, should take two pandas series and return a value (Float64)
        :return: scores
            Pandas series of scores for each gene model
        """
        # get prediction of gene expression
        predicted = self.predict(tf_expression=tf_expression)
        print(f"Predicted: \n{predicted}")
        # Make sure prediction and gene expression have the same index and columns
        predicted = predicted.loc[gene_expression.index, gene_expression.columns]
        # Score the models using the provided metric
        scores = df_reduce(gene_expression, predicted, metric, axis=1)
        return scores

    def create_exogenous_array(self, gene: str, data: pd.DataFrame, intercept: bool = True) -> pd.DataFrame:
        """
        Function to create the exogenous array for gene based on TF targets from the data dataframe
        :param intercept: Whether the model should fit with intercepts
        :param gene: Gene for which to create the exogenous array
        :param data: Expression data, dimensions of (n_samples, n_genes)
        :return: exog_array
            The exogenous array for gene
        """
        exog_array = data[self.gene_to_tf_dict[gene]].copy()
        if intercept:
            exog_array["intercept"] = 1.
        return exog_array


class TfKinaseRestrictedModel:
    def __init__(self, kinase_dict: dict, tf_dict: dict, gene_list: list):
        """
        init Function for TF Only Model
        :param kinase_dict: dict of kinase:gene showing the phosphorylation targets
        :param tf_dict: dict of tf:gene showing tf targets
        :param gene_list: Total list of genes which will be in the compendia
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
        # Variable for if the models have intercepts
        self.intercept = None
        # Variable for if the model is regularized
        self.regularized = None

    def fit(self, data: pd.DataFrame, regularized: bool, intercept: bool = True, verbose: bool = False, **kwargs):
        """
        Function to fit the model to the provided data
        :param data: Gene expression data, including targeted genes, kinases, and tfs
        :param regularized: Whether regularization is desired
        :param intercept: Whether an intercept is desired
        :param verbose: Whether a verbose output is desired
        :param kwargs: Keyword arguments passed to statsmodels OLS fit method, either fit or fit_regularized
        :return: None
        """
        # Create variable for progress bar, used if verbose output is desired
        bar = None
        # Create the dataframe to hold the model coefficients
        self.model_coefficients = pd.DataFrame(0., index=self.targeted_genes, columns=self.tf_list + self.kinase_list)
        # Add intercept if needed,
        if intercept:
            self.intercept = True
            self.model_coefficients["intercept"] = 0.
        # Check if verbose output is desired, and setup progress bar if needed
        if verbose:
            print("Finding Model Coefficients")
            bar = progress_bar.ProgressBar(total=len(self.targeted_genes), divisions=10)
        # Iterate through the genes and find the model coefficients
        for gene in self.targeted_genes:
            if verbose:
                bar.inc()
            # Create the exogenous array
            exog = self.create_exogenous_array(gene, data).dropna(axis=0)
            # Create the endogenous array
            endog = data[gene].dropna(axis=0)
            # determine which samples are fully defined
            defined_samples = lh.find_intersect(list(exog.index), list(endog.index))
            # modify endog and exog to only include the defined samples
            endog = endog[defined_samples]
            exog = exog.loc[defined_samples]
            # Create the model for the gene
            model = sm.OLS(endog=endog, exog=exog)
            # Fit the model, with regularization if that is desired
            if regularized:
                self.regularized = True
                results = model.fit_regularized(**kwargs)
            else:
                results = model.fit()
            # Get the coefficients from the fit model
            coefficients = results.params
            # Find which tfs and kinases are in the model
            tfs = self.gene_to_tf_dict[gene]
            kinases = self.gene_to_tf_to_kinase_dict[gene]
            # Add the coefficients to the coefficient array
            for tf in tfs:
                self.model_coefficients.loc[gene, tf] = coefficients[tf]
            for kinase in kinases:
                self.model_coefficients.loc[gene, kinase] = coefficients[kinase]
            if self.intercept:
                self.model_coefficients.loc[gene, "intercept"] = coefficients["intercept"]
        if verbose:
            print("Found Coefficients")

    def predict(self, tf_expression, kinase_expression):
        """
        Function to predict gene expression based on TF expression
        :param kinase_expression: Dataframe of kinase expression, dimensions are (n_samples, n_kinases)
        :param tf_expression: Dataframe of tf expression, dimensions are (n_samples, n_tfs)
        :return: pred_gene_expression
            Dataframe with predicted gene expression, dimensions are (n_samples, n_genes)
        """
        expression = pd.concat([tf_expression, kinase_expression], axis=1)
        input_df = expression[self.model_coefficients.columns]
        output = pd.DataFrame(
            np.transpose(
                np.matmul(self.model_coefficients.to_numpy(),
                          np.transcpose(input_df.fillna(0).to_numpy())
                          )
            ), index=expression.index, columns=self.model_coefficients.index
        )
        return output

    def score(self,
              tf_expression: pd.DataFrame,
              kinase_expression,
              gene_expression: pd.DataFrame,
              metric: typing.Callable):
        """
        Method to score the model against the provided gene_expression values
        :param kinase_expression: Kinase expression dataframe, dimensions are (n_samples, n_kinases)
        :param tf_expression: TF expression dataframe, dimensions are (n_samples, n_tfs)
        :param gene_expression: Gene expression dataframe, dimensions are (n_samples, n_genes)
        :param metric: Metric to use for scoring, should take two pandas series and return a value (Float64)
        :return: scores
            Pandas series of scores for each gene model
        """
        # get prediction of gene expression
        predicted = self.predict(tf_expression=tf_expression, kinase_expression=kinase_expression)
        print(f"Predicted: \n{predicted}")
        # Make sure prediction and gene expression have the same index and columns
        predicted = predicted.loc[gene_expression.index, gene_expression.columns]
        # Score the models using the provided metric
        scores = df_reduce(gene_expression, predicted, metric, axis=1)
        return scores

    def create_exogenous_array(self, gene, data) -> pd.DataFrame:
        """
        Function to create the exogenous array for model fitting
        :param gene: Gene to create the array for
        :param data: Data to create the array from
        :return: exog
            pandas dataframe representing the exogenous array
        """
        # Find which tfs and kinases should be included in the model
        tfs = self.gene_to_tf_dict[gene]
        kinases = self.gene_to_tf_to_kinase_dict[gene]
        # Create the exogenous array
        exog = data[tfs + kinases].copy()
        # If an intercept is needed, add a column of all 1s
        if self.intercept:
            exog["intercept"] = 1.
        # Return the exogenous array
        return exog


class TfKinaseFullModel:
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

    def fit(self, data: pd.DataFrame, regularized: bool, intercept: bool = True, verbose: bool = False, **kwargs):
        """
        Function to fit the model to the provided data
        :param data:
        :param regularized:
        :param intercept:
        :param verbose:
        :param kwargs:
        :return:
        """
        data_columns = list(data.columns)
        full_data = data.copy()
        interaction_terms_full = []
        interaction_series_list = []
        intercept_series = pd.Series(1., index=full_data.index)
        intercept_series.name = "intercept"
        # Add all possible interaction terms and intercept to the gene expression dataframe
        for tf in self.tf_list:
            # If the TF is in the provided data
            if tf in data_columns:
                # find the kinases that target the given tf, and add interaction terms for them
                for kinase in self.gene_to_kinase_dict[tf]:
                    if kinase in data_columns:
                        interaction_series_list.append((full_data[kinase] * full_data[tf]))
                        interaction_series_list[-1].name = f"{kinase}_{tf}"
                        interaction_terms_full.append(f"{kinase}_{tf}")
        full_data = pd.concat([full_data] + interaction_series_list + [intercept_series], axis=1)
        # If intercept is desired, add it to the full_data,
        if intercept:
            self.intercept = intercept
        # Create the dataframe to hold the model coefficients
        column_list = self.tf_list + self.kinase_list + interaction_terms_full
        self.model_coefficients = pd.DataFrame(0., index=self.targeted_genes, columns=column_list)
        # if a verbose output is desired, create the progress bar object
        if verbose:
            print("Finding model coefficients")
            bar = progress_bar.ProgressBar(total=len(self.targeted_genes), divisions=10)
        else:
            bar = None
        # Iterate through all the genes and fit the models
        for gene in self.targeted_genes:
            if verbose:
                bar.inc()
            # Find the exogenous array
            exog = self.create_exogenous_array(gene=gene, data=full_data).dropna(axis=0)
            # Find the endogenous array
            endog = full_data[gene].dropna(axis=0)
            # Make sure that the exog and endog only include fully defined samples
            defined_samples = lh.find_intersect(list(endog.index), list(exog.index))
            endog = endog[defined_samples]
            exog = exog.loc[defined_samples]
            # Create the model object
            model = sm.OLS(endog=endog, exog=exog)
            # Fit the model
            if regularized:
                self.regularized = regularized
                results = model.fit_regularized(**kwargs)
            else:
                self.regularized = not regularized
                results = model.fit(**kwargs)
            # Pull out the coefficients
            coefficients = results.params
            # Add the coefficients to the model_coefficients array
            for coefficient in coefficients.index:
                self.model_coefficients.loc[gene, coefficient] = coefficients[coefficient]
        if verbose:
            print("Found Coefficients")

    def predict(self, tf_expression: pd.DataFrame, kinase_expression: pd.DataFrame) -> pd.DataFrame:
        """
        Function to predict gene expression based on TF expression
        :param kinase_expression: Dataframe of kinase expression, dimensions are (n_samples, n_kinases)
        :param tf_expression: Dataframe of tf expression, dimensions are (n_samples, n_tfs)
        :return: pred_gene_expression
            Dataframe with predicted gene expression, dimensions are (n_samples, n_genes)
        """
        expression = pd.concat([tf_expression, kinase_expression], axis=1)
        for tf in self.tf_list:
            for kinase in self.gene_to_kinase_dict[tf]:
                expression[f"{kinase}_{tf}"] = expression[kinase] * expression[tf]
        input_df = expression[self.model_coefficients.columns]
        output = pd.DataFrame(
            np.transpose(
                np.matmul(self.model_coefficients.to_numpy(),
                          np.transcpose(input_df.fillna(0).to_numpy())
                          )
            ), index=expression.index, columns=self.model_coefficients.index
        )
        return output

    def score(self,
              tf_expression: pd.DataFrame,
              kinase_expression,
              gene_expression: pd.DataFrame,
              metric: typing.Callable) -> pd.Series:
        """
        Method to score the model against the provided gene_expression values
        :param kinase_expression: Kinase expression dataframe, dimensions are (n_samples, n_kinases)
        :param tf_expression: TF expression dataframe, dimensions are (n_samples, n_tfs)
        :param gene_expression: Gene expression dataframe, dimensions are (n_samples, n_genes)
        :param metric: Metric to use for scoring, should take two pandas series and return a value (Float64)
        :return: scores
            Pandas series of scores for each gene model
        """
        # get prediction of gene expression
        predicted = self.predict(tf_expression=tf_expression, kinase_expression=kinase_expression)
        print(f"Predicted: \n{predicted}")
        # Make sure prediction and gene expression have the same index and columns
        predicted = predicted.loc[gene_expression.index, gene_expression.columns]
        # Score the models using the provided metric
        scores = df_reduce(gene_expression, predicted, metric, axis=1)
        return scores

    def create_exogenous_array(self, gene: str, data: pd.DataFrame) -> pd.DataFrame:
        """
        Function to create exogenous array for model fitting
        :param gene: Gene to create exogenous array for
        :param data: Data to use for the creation of the exogenous array
        :return: exog
            The exogenous array, a pandas dataframe
        """
        # Create a list of columns for the exogenous array
        exog_columns = self.gene_to_tf_dict[gene] + self.gene_to_tf_to_kinase_dict[gene]
        # Find the interaction terms which need to be added,
        interaction_terms = self.find_interaction_terms(gene)
        # Iterate through the interaction terms, and add them to the columns list
        for kinase, tf in interaction_terms:
            inter_repr = f"{kinase}_{tf}"
            exog_columns.append(inter_repr)
        if self.intercept:
            exog_columns.append("intercept")
        exog = data[exog_columns]
        return exog

    def find_interaction_terms(self, gene: str) -> list[tuple]:
        """
        Function to find all the interaction terms for a given gene
        :param gene: Gene to find the interaction terms for
        :return: interaction_terms_list
            list of interaction terms for the gene
        """
        # Create an empty list to hold the interaction term tuples
        interaction_terms_list = []
        # Find which tfs target the gene
        tfs = self.gene_to_tf_dict[gene]
        # for each tf, find the kinases that target them, and add interaction terms tuple to the list
        for tf in tfs:
            kinases = self.gene_to_kinase_dict[tf]
            for kinase in kinases:
                interaction_terms_list.append((kinase, tf))
        return interaction_terms_list
