"""
Module containing utility functions for working with the RNA seq data
"""
# External Imports
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
