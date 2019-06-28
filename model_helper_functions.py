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
    # if 1 == 1:
    if 1 == len(smaller):
        return [[x] for x in np.linspace(start=0, stop=larger - 1, num=smaller[0], dtype=np.int32)]
    else:
        return [[x, y] for x in np.linspace(start=0, stop=larger - 1, num=smaller[0], dtype=np.int32)
                for y in np.linspace(start=0, stop=larger - 1, num=smaller[1], dtype=np.int32)]


def normalize(v):
    mu = np.nanmean(v)
    sigma = np.nanstd(v)

    return (v - np.nanmean(v)) / np.nanstd(v), mu, sigma


def undo_normalize(v, mu, sigma):
    return mu + (sigma * v)


def quadratic_finite_difference(v):
    return tf.reduce_sum([(v[i] - (2 * v[i - 1]) + v[i - 2]) ** 2 for i in range(2, v.shape.as_list()[0])])


def validate_and_save_inputs(model, covariates, spline_count, x_vars, y_var):
    # Check the covariate list for valid data, and save it accordingly.
    if isinstance(covariates, dict) and 1 <= len(covariates.keys()) <= 2:
        for i, key in enumerate(covariates.keys()):
            if not isinstance(key, str):
                raise TypeError(f"Covariate names must be strings. Covariate {i} was not.")

        setattr(model, "covariates", covariates)
        setattr(model, "data_shape", [len(covariates[key]) for key in covariates.keys()])
        if 1 == len(covariates.keys()):
            setattr(model, "dimension", 1)
        else:
            setattr(model, "dimension", 2)

    elif isinstance(covariates, dict):
        raise AttributeError(f"This package only supports 1-or-2-dimensional models. "
                             f"You supplied {len(covariates.keys())} covariates.")
    else:
        raise TypeError("Covariates must be supplied as a dictionary with covariate names "
                        "as strings for keys, and covariate values for values.")

    # Validate the spline count supplied, and make sure its dimensions coincide with the covariate dimensions.
    if isinstance(spline_count, tuple) and len(spline_count) != 2:
        raise AttributeError(f"This package only supports 1-or-2-dimensional data. "
                             f"You supplied {len(spline_count)} dimensions for the spline counts.")
    elif isinstance(spline_count, tuple) and len(spline_count) != len(covariates):
        raise AttributeError(f"{len(covariates)} were supplied, but only {len(spline_count)} spline counts. "
                             f"Must have one spline count per dimension.")
    elif isinstance(spline_count, tuple) and len(spline_count) == len(covariates):
        for i, c in enumerate(spline_count):
            if not isinstance(c, int):
                raise TypeError(f"Counts must be integers, but count {i} was not.")

        setattr(model, "spline_counts", list(spline_count))
        setattr(model, "spline_total", spline_count[0] * spline_count[1])
    elif isinstance(spline_count, int) and 1 == len(covariates):
        setattr(model, "spline_counts", [spline_count])
        setattr(model, "spline_total", spline_count)
    elif isinstance(spline_count, int) and 2 == len(covariates):
        setattr(model, "spline_counts", [spline_count, spline_count])
        setattr(model, "spline_total", spline_count ** 2)
    else:
        raise AttributeError("Spline count must either be a single positive integer, "
                             "or a tuple containing two positive integers.")

    # Validate the names of the x-vars.
    if isinstance(x_vars, list):
        for i, x in enumerate(x_vars):
            if not isinstance(x, str):
                raise TypeError(f"X variables must be strings: {i} was not.")

        setattr(model, "x_vars", x_vars)
    elif isinstance(x_vars, str):
        setattr(model, "x_vars", [x_vars])
    else:
        raise AttributeError("X variable names must be passed as a list of strings, or a single string")

    if isinstance(y_var, str) or y_var is None:
        setattr(model, "y_var", y_var)
    else:
        raise AttributeError("Y variable name must be strings")
