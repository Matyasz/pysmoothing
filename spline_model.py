import model_helper_functions as mhf
from splines import create_splines

import tensorflow as tf


class SplineModel:
    def __init__(self, covariates, spline_count, x_vars, y_var: str=None,
                 penalize: bool=True, resolution: int=500, seed: int=0):
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
        self.indicators = {x: [] for x in getattr(self, "x_vars") + [y_var]}
        self.resolution = resolution

    def make(self):
        self._make_placeholders()
        self._make_coefficients()
        self._make_model()
        self._make_objective_functions()

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

        # Make the placeholders for the penalty values
        for var in getattr(self, 'x_vars'):
            for cov in getattr(self, 'covariates'):
                ph_name = f"{var}_{cov}_penalty"
                phs[ph_name] = tf.placeholder(dtype=tf.float32,
                                              shape=[1], name=ph_name)

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
                coeff[var] = tf.Variable(initial_value=tf.random_normal([getattr(self, "spline_total") + 1],
                                                                        mean=0.0, stddev=0.001),
                                         dtype=tf.float32,
                                         name=f"{var}_coefficients")
            else:
                coeff[var] = tf.Variable(initial_value=tf.random_normal([getattr(self, "spline_total") + 1],
                                                                        mean=0.0, stddev=0.001),
                                         dtype=tf.float32,
                                         name=f"{var}_coefficients")

        setattr(self, "coefficients", coeff)

    def _make_model(self):
        models = {}

        # Construct the splines for use in the model
        # If we have a two dimensional model, create the 2D splines and use them
        if 1 == getattr(self, "dimension"):
            model_splines = tf.transpose(tf.constant(create_splines(getattr(self, "spline_total"),
                                                                    resolution=self.resolution),
                                                     dtype=tf.float32))

            sum_axis = 1
        else:
            from splines import tensor_product
            s1 = create_splines(getattr(self, "spline_counts")[0])
            s2 = create_splines(getattr(self, "spline_counts")[1])

            model_splines = tensor_product(s1, s2, include_const_layer=True)

            sum_axis = 2

        # Construct the model for each variable
        for var in getattr(self, 'x_vars'):
            model = tf.reduce_sum(tf.multiply(model_splines, getattr(self, "coefficients").get(var)),
                                  axis=sum_axis, name=f"{var}_model")
            models[var] = model

        # If there is a Y variable for the model, create that model
        if isinstance(getattr(self, "y_var"), str):
            model = tf.constant(0., dtype=tf.float32)
            for m in models.keys():
                model = model + models[m]

            models[getattr(self, "y_var")] = model
        setattr(self, "models", models)

    def _make_penalties(self):
        pass

    def _make_objective_functions(self):
        objectives = {}

        # Define the objective functions for all X variable models.
        # Often, the splines will have far more input values than the actual data, so we need to make
        # sure that we grab the correct number of evenly spaced spline outputs for this calculation.
        for var in getattr(self, 'x_vars'):
            obj = tf.reduce_sum(tf.square(tf.subtract(
                tf.gather(getattr(self, 'models').get(var),
                          indices=mhf.match_indices(larger=self.resolution,
                                                    smaller=getattr(self, "data_shape")[0])),
                getattr(self, 'placeholders').get(var))),
                                name=f"{var}_objective")

            objectives[var] = obj

        # If there is a Y variable, define its objective function.

        setattr(self, "objectives", objectives)

    def _make_optimizers(self):
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

            setattr(self, "spline_counts", list(spline_count))
            setattr(self, "spline_total", spline_count[0] * spline_count[1])
        elif isinstance(spline_count, int) and 1 == len(covariates):
            setattr(self, "spline_counts", [spline_count])
            setattr(self, "spline_total", spline_count)
        elif isinstance(spline_count, int) and 2 == len(covariates):
            setattr(self, "spline_counts", [spline_count, spline_count])
            setattr(self, "spline_total", spline_count ** 2)
        else:
            raise AttributeError("Spline count must either be a single positive integer, "
                                 "or a tuple containing two positive integers.")

        # Validate the names of the x-vars.
        if isinstance(x_vars, list):
            for i, x in enumerate(x_vars):
                if not isinstance(x, str):
                    raise TypeError(f"X variables must be strings: {i} was not.")

            setattr(self, "x_vars", x_vars)
        elif isinstance(x_vars, str):
            setattr(self, "x_vars", [x_vars])
        else:
            raise AttributeError("X variable names must be passed as a list of strings, or a single string")

        if isinstance(y_var, str) or y_var is None:
            setattr(self, "y_var", y_var)
        else:
            raise AttributeError("Y variable name must be strings")
