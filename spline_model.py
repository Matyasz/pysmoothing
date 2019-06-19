from splines import create_splines

import tensorflow as tf


class SplineModel:
    def __init__(self, covariates, spline_count, x_vars: list, y_var: str=None, penalize: bool=True, seed: int=0):
        """
            Constructor only takes the basic info necessary to define the model,
            then methods will be available for the user to crete the model,
            fit the model to a particular data set, or add additional indicators.

            :param covariates:
            :param spline_count:
            :param x_vars:
            :param y_var:
            :param seed:
        """
        tf.set_random_seed(seed)
        self.penalize = penalize
        self.validate_and_save_inputs(covariates, spline_count, x_vars, y_var)
        self.indicators = {x: [] for x in x_vars + [y_var]}

    def fit(self, **kwargs):
        data = {}
        for var, value in kwargs:
            data[var] = value

        with tf.Session() as sess:
            sess.run(tf.initialize_all_variables)

    def predict(self, vars=None):
        pass

    def _make_placeholders(self):
        """
            Make the placeholder objects for the data that will be passed to the fit function by the user.
        """
        phs = {}

        # First get the total number of penalties needed.
        penalty_count = getattr(self, "spline_count")[0]
        if 2 == getattr(self, "dimension"):
            penalty_count *= getattr(self, "spline_count")[1]

        # Make the placeholders for the penalty values
        for var in getattr(self, 'x_vars'):
            for cov in getattr(self, 'covariates'):
                ph_name = f"{var}_{cov}_penalty"
                phs[ph_name] = tf.placeholder(dtype=tf.float32,
                                              shape=getattr(self, "data_shape"), name=ph_name)

        # Make the placeholders for the data to be fitted
        for var in getattr(self, 'x_vars') + [getattr(self, 'y_var')]:
            ph_name = f"{var}_placeholder"
            phs[var] = tf.placeholder(dtype=tf.float32,
                                      shape=getattr(self, "data_shape"), name=ph_name)

        setattr(self, "placeholders", phs)

    def _make_coefficients(self):
        coeff = {}

        for var in getattr(self, "x_vars"):
            if 1 == getattr(self, "dimension"):
                coeff[var] = tf.Variable(initial_value=tf.random_normal(getattr(self, "spline_counts") + 1,
                                                                        mean=0.0, stddev=0.001),
                                         dtype=tf.float32,
                                         name=f"{var}_coefficients")

        setattr(self, "coefficients", coeff)

    def _make_model(self):
        models = {}

        # Construct the splines for use in the model
        # If we have a two dimensional model, create the 2D splines and use them
        if 1 == getattr(self, "dimension"):
            model_splines = create_splines(getattr(self, "spline_count") + 1)
        else:
            from splines import tensor_product
            s1 = create_splines(getattr(self, "spline_count")[2])
            s2 = create_splines(getattr(self, "spline_count")[1])

            model_splines = tensor_product(s1, s2, include_const_layer=True)

        # Construct the model for each variable
        for var in getattr(self, 'x_vars'):
            pass

        # If there is  aY variable for the model, create that model
        if isinstance(getattr(self, "y_var"), str):
            pass

        setattr(self, "models", models)

    def _make_penalty(self):
        pass

    def _make_objective_function(self):
        pass

    def _make_optimizer(self):
        pass

    def add_indicator(self, **kwargs):
        pass

    def validate_and_save_inputs(self, covariates, spline_count, x_vars, y_var):
        # Check the covariate list for valid data, and save it accordingly.
        if isinstance(covariates, dict) and 1 <= len(covariates.keys()) <= 2:
            for i, key in enumerate(covariates.keys()):
                if not isinstance(key, str):
                    raise TypeError(f"Covariate names must be strings. Covariate {i} was not.")

            setattr(self, "covariates", covariates)
            setattr(self, "data_shape", [len(covariates[key]) for key in covariates.keys()])
            if 1 == len(covariates.keys()):
                setattr(self, "dimension", 1)
            else:
                setattr(self, "dimension", 2)

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

            setattr(self, "spline_count", list(spline_count))
        elif isinstance(spline_count, int) and 1 == len(covariates):
            setattr(self, "spline_count", [spline_count])
        else:
            raise AttributeError("Spline count must either be a single positive integer, "
                                 "or a tuple containing two positive integers.")

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
