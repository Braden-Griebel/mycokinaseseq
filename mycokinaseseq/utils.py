"""
Module containing utility functions for working with the RNA seq data
"""
# Core Imports
import random
import warnings
# External Imports
import numpy as np
import pandas as pd


def rpkm_to_tpm(rpkm_df: pd.DataFrame) -> pd.DataFrame:
    """
    Function to convert data in RPKM (reads per kilobase million) form to TPM (transcripts per million) form
        source: Zhao S, Ye Z, Stanton R. Misuse of RPKM or TPM normalization when comparing across samples and
            sequencing protocols. RNA. 2020 Aug;26(8):903-909. doi: 10.1261/rna.074922.120. Epub 2020 Apr 13.
            PMID: 32284352; PMCID: PMC7373998.
    :param rpkm_df: Dataframe with genes as columns, and samples as rows in the form of RPKM
    :return: tpm_df
        Dataframe of the same dimensions as the input, converted to TPM form
    """
    gene_list = list(rpkm_df.columns)
    sample_list = list(rpkm_df.index)
    rpkm_df["sum"] = rpkm_df.sum(axis=1)
    tpm_df = rpkm_df[gene_list].div(rpkm_df["sum"], axis=0) * 10 ** 6
    return tpm_df


def permute_labels_total(dataframe: pd.DataFrame, axis: int, preserve_order: bool = False) -> pd.DataFrame:
    """
    Function to permute all the labels for a dataframe along given axis without modifying the underlying data. Used for
    creating a null model.
    :param preserve_order: Boolean, whether the order of labels from the original dataframe should be preserved
    :param dataframe: Dataframe to permute models on
    :param axis: Which axis to permute the labels of
    :return: permuted_dataframe
        Pandas dataframe with permuted labels
    """
    permuted_dataframe = dataframe.copy(deep=True)
    if axis == 0:
        rows = list(dataframe.index)
        random.shuffle(rows)
        permuted_dataframe.index = pd.Index(rows)
        if (preserve_order):
            original_rows = list(dataframe.index)
            permuted_dataframe = permuted_dataframe.reindex(original_rows, axis=axis)
    elif axis == 1:
        columns = list(dataframe.columns)
        random.shuffle(columns)
        permuted_dataframe.columns = pd.Index(columns)
        if (preserve_order):
            original_columns = list(dataframe.columns)
            permuted_dataframe = permuted_dataframe.reindex(original_columns, axis=axis)
    return permuted_dataframe


def permute_labels_subset(dataframe, axis, subset, preserve_order: bool = False):
    """
    Function to permute a subset of all the labels for a dataframe along a given axis without modifying the underlying
        data. Used for creating a null model.
    :param preserve_order: Boolean, whether order of labels from the original data should be preserved
    :param dataframe: Dataframe to shuffle the labels for
    :param axis: Axis to perform the shuffling on
    :param subset: The subset of the axis labels to permute
    :return: permuted_dataframe
        Copy of the dataframe with shuffled labels
    """
    if axis == 0:
        not_subset = [x for x in dataframe.index if x not in subset]
        if not set(subset).issubset(set(dataframe.index)):
            warnings.warn("Not all of subset values are in dataframe, dropping values")
            subset = [x for x in subset if x in list(dataframe.index)]
        subset_df = permute_labels_total(dataframe.loc[subset], axis=axis)
        not_subset_df = dataframe.loc[not_subset].copy()
        permuted_df = pd.concat([not_subset_df, subset_df], axis=0)
        if preserve_order:
            # Forcing copy of the index
            original_index = pd.Index(list(dataframe.index))
            permuted_df = permuted_df.reindex(original_index, axis=0)
    elif axis == 1:
        not_subset = [x for x in dataframe.columns if x not in subset]
        if not set(subset).issubset(set(dataframe.columns)):
            warnings.warn("Not all of subset values are in dataframe, dropping values")
            subset = [x for x in subset if x in list(dataframe.columns)]
        subset_df = permute_labels_total(dataframe[subset], axis=axis)
        not_subset_df = dataframe[not_subset].copy()
        permuted_df = pd.concat([not_subset_df, subset_df], axis=1)
        if preserve_order:
            # Forcing a copy
            original_columns = pd.Index(list(dataframe.columns))
            permuted_df = permuted_df.reindex(original_columns, axis=1)
    return permuted_df
