"""
Module containing utility functions for working with the RNA seq data
"""
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


def permute_labels_total(dataframe, axis):
    """
    Function to permute all the labels for a dataframe along given axis without modifying the underlying data. Used for
    creating a null model.
    :param dataframe: Dataframe to permute models on
    :param axis: Which axis to permute the labels of
    :return: permuted_dataframe
        Pandas dataframe with permuted labels
    """
    if axis == 0:
        permuted_dataframe = dataframe.reindex(np.random.permutation(dataframe.index), axis=axis)
    elif axis == 1:
        permuted_dataframe = dataframe.reindex(np.random.permutation(dataframe.columns), axis=axis)
    return permuted_dataframe


def permute_labels_subset(dataframe, axis, subset):
    """
    Function to permute a subset of all the labels for a dataframe along a given axis without modifying the underlying
        data. Used for creating a null model.
    :param dataframe: Dataframe to shuffle the labels for
    :param axis: Axis to perform the shuffling on
    :param subset: The subset of the axis labels to permute
    :return: permuted_dataframe
        Copy of the dataframe with shuffled labels
    """
    if axis == 1:
        not_subset = [x for x in dataframe.columns if x not in subset]
        subset_df = permute_labels_total(dataframe[subset], axis=axis)
        not_subset_df = dataframe[not_subset].copy()
        permuted_df = pd.concat([not_subset_df, subset_df], axis=1)
    elif axis == 0:
        not_subset = [x for x in dataframe.index if x not in subset]
        subset_df = permute_labels_total(dataframe.loc[subset], axis=axis)
        not_subset_df = dataframe.loc[not_subset].copy()
        permuted_df = pd.concat([not_subset_df, subset_df], axis=0)
    return permuted_df
