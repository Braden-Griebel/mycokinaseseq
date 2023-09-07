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


def list_contains_duplicate(list1: list) -> bool:
    """
    Determing if the list contains duplicate items
    :param list1: List to check duplicates for
    :return: duplicates
        Bool, True if there are duplicates
    """
    # Check if the length of unique elements is the same as the length of the original list
    if len(list1) != len(list(np.unique(list1))):
        # If it isn't there is a duplicate
        return True
    return False
