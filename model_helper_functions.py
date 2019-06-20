import numpy as np


def match_indices(larger, smaller):
    """
        Take two tensor sizes, and determine how to "shrink" the larger one to make it the same
        size as the smaller one. Basically, evenly reduce the resolution of the larger tensor
        so it can be compared to the smaller tensor.

        :param larger: The size of the larger tensor
        :param smaller: The size of the smaller tensor
        :return: The indices from the larger tensor that you should use to compare the two tensors
    """

    return [x for x in np.linspace(start=0, stop=larger - 10e-4, num=smaller, dtype=np.int32)]
