import numpy as np
import tensorflow as tf


def match_indices(larger, smaller):
    """
        Take two tensor sizes, and determine how to "shrink" the larger one to make it the same
        size as the smaller one. Basically, evenly reduce the resolution of the larger tensor
        so it can be compared to the smaller tensor.

        :param larger: The size of the larger tensor
        :param smaller: The size of the smaller tensor
        :return: The indices from the larger tensor that you should use to compare the two tensors
    """

    return [x for x in np.linspace(start=0, stop=larger - 1, num=smaller, dtype=np.int32)]


def normalize(v):
    return (v - np.nanmean(v)) / np.nanstd(v)


def quadratic_finite_difference(v):
    return tf.reduce_sum([(v[i] - (2 * v[i - 1]) + v[i - 2]) ** 2 for i in range(2, v.shape.as_list()[0])])
