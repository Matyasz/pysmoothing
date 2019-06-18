from splines import create_splines

import tensorflow as tf


class SplineModel:
    def __init__(self, covariates, spline_count, x_vars: list, y_var: str=None, seed: int=0):
        """
            Constructor only takes the basic info necessary to define the model,
            then methods will be available for the user to crete the model,
            fit the model to a particular data set, or add additional indicators.

            :param x_vars:
            :param y_var:
            :param spline_count:
            :param seed:
        """
        tf.set_random_seed(seed)
        self.validate_and_save_inputs(covariates, spline_count, x_vars, y_var)
        self.indicators = {x: [] for x in x_vars + [y_var]}

    def validate_and_save_inputs(self, covariates, spline_count, x_vars, y_var):
        # Check the covariate list for valid data, and save it accordingly.
        if covariates is None:
            raise AttributeError("Covariate list cannot be empty. Must name all dimensions.")
        elif isinstance(covariates, list) and len(covariates) <= 2:
            setattr(self, "covariates", covariates)

            for i, cov in enumerate(covariates):
                if not isinstance(cov, str):
                    raise TypeError(f"Covariate names must be strings. Covariate {i} was not.")

        elif isinstance(covariates, list) and len(covariates) > 2:
            raise AttributeError(f"This package only supports up to two dimensional data sets. "
                                 f"The data you supplied was {len(covariates)}-dimensional.")
        elif isinstance(covariates, str):
            setattr(self, "covariates", [covariates])
        else:
            raise AttributeError("Covariates supplied must either be a string (for one covariate),"
                                 "or a list of length either 1 or 2 with string entries.")

        # Validate the spline count supplied, and make sure its dimensions coincide with the covariate dimensions.
        if isinstance(spline_count, tuple) and len(spline_count) != 2:
            raise AttributeError(f"This package only supports 1-or-2-dimensional data. "
                                 f"You supplied {len(spline_count)} dimensions for the spline counts.")
        elif isinstance(spline_count, tuple) and len(spline_count) != len(covariates):
            raise AttributeError(f"{len(covariates)} were supplied, but only {len(spline_count)} spline counts. "
                                 f"Must have one spline count per dimension.")
        else:
            setattr(self, "spline_count", spline_count)

        # Validate the names of the x-vars.
        if isinstance(x_vars, list):
            for i, x in enumerate(x_vars):
                if not isinstance(x, str):
                    raise TypeError(f"X variables must be strings: {i} was not.")

            setattr(self, "x_vars", x_vars)
        else:
            raise AttributeError("X variable names must be strings")

        if isinstance(y_var, str) or y_var is None:
            setattr(self, "y_var", y_var)
        else:
            raise AttributeError("Y variable name must be strings")
