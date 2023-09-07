# Core Imports
from typing import Union, Iterable
# External Imports
import pandas as pd
import qnorm


# Local imports

def class_qnorm(data, classes: Union[Iterable, dict]):
    """
    Perform class based quantile normalization
    :param data: data to normalize, columns are features, and rows are samples
    :param classes: Which samples belong to which class, can be list of classes (order corresponding to index
        of data), or a dict of class:[list of indices corresponding to class]
    :return: normalized_data
        Pandas dataframe following quantile normalization
    """
    if classes is not dict:
        # Check if classes can be iterated through
        if classes is not Iterable:
            raise ValueError("Classes is not dict, or iterable")
        # Create dict to hold class:index information
        class_dict = {}
        # Iterate through the classes, match them to the index of the dataframe, and create fill in the dict
        for index, class_name in zip(data.index, classes):
            if class_name in class_dict:
                class_dict[class_name].append(index)
            else:
                class_dict[class_name] = [index]
    else:
        class_dict = classes
    normalized_data_list = []
    # Iterate through all the classes, and perform the quantile normalization
    for class_name, index_list in class_dict.items():
        # If there is only one sample in this class, no quantile normalization can be performed, just append to the list
        if len(index_list) == 1:
            normalized_data_list.append(data.loc[index_list])
        elif len(index_list) > 1:
            normalized_data_list.append(qnorm.quantile_normalize(data.loc[index_list], axis=0))
    # Concatenate together the normalized data
    normalized_data = pd.concat(normalized_data_list, axis=0)
    return normalized_data

