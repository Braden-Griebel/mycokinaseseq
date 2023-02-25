"""
Functions to help with list processing
"""
# Imports
import numpy as np


# Create a helper function to return a list of unique elements from a list
def get_unique_list(list_in: list) -> list:
    """
    Find unique elements in a list
    :param list_in: list (or other array type
    :return: list with only the unique elements from list_in
    """
    return list(np.unique(list_in))


# Find the intersection of two lists
def find_intersect(list1: list, list2: list) -> list:
    """
    Find elements in both lists
    :param list1:
    :param list2:
    :return: List with the elements found in both lists
    """
    return [value for value in list1 if value in list2]
