import model_helper_functions as mhf
from splines import create_splines

from numpy.linalg import norm
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

        mhf.validate_and_save_inputs(self, covariates, spline_count, x_vars, y_var)

        self.indicators = {x: [] for x in getattr(self, "x_vars") + [y_var]}

        self.penalize = penalize
        self.resolution = resolution
        self.global_step = tf.Variable(0, trainable=False, name="global_step")

    def make(self):
        self._make_placeholders()
        self._make_coefficients()
        self._make_model()
        self._make_penalties()
        self._make_objective_functions()
        self._make_optimizers()

    def fit_one(self, var: str, raw_data, tolerance: float=0.01, print_freq: int=500):
        """
            Used to fit one X variable and return the data.

            :param var:
            :param raw_data:
            :param tolerance:
            :param print_freq:
            :return:
        """
        data, mean, stdev = mhf.normalize(raw_data)
        prediction = self._fit(var, data, tolerance, print_freq)

        prediction = mhf.undo_normalize(prediction[0], mean, stdev)
        return prediction

    def fit_multi(self, fit_vars: list, raw_data, tolerance: float=0.01, print_freq: int=500):
        """
            Used to fit all variables, and save a dictionary to the object which stores the fitted values.
        """
        predictions = {}

        for var in fit_vars:
            prediction = self._fit(var, raw_data, tolerance, print_freq)
            predictions[var] = prediction

        return predictions

    def _fit(self, var, data, tol, pr_freq):
        """
            Holds the logic for fitting a model to a variable.
        """
        data = {getattr(self, "placeholders").get(var): data}
        gradients = getattr(self, "optimizers").get(var).compute_gradients(getattr(self, "objectives").get(var),
                                                                           getattr(self, "coefficients").get(var))
        minimizer = getattr(self, "optimizers").get(var).apply_gradients(gradients, global_step=self.global_step)
        counter = 0

        print(f"Fitting {var}")
        with tf.Session() as sess:
            sess.run(tf.global_variables_initializer())

            # Print the starting values
            grad, loss = sess.run([gradients[0][0], getattr(self, "objectives").get(var)], feed_dict=data)
            grad = norm(grad)
            print(f"Iterations: {counter}   loss: {loss}   grad: {grad}")

            while grad > tol:
                _, grad, loss = sess.run([minimizer, gradients[0][0], getattr(self, "objectives").get(var)],
                                         feed_dict=data)
                grad = norm(grad)
                counter += 1

                if 0 == counter % pr_freq:
                    print(f"Iterations: {counter}   loss: {loss}   grad: {grad}")
            # Print the final values
            print(f"Iterations: {counter}   loss: {loss}   grad: {grad}")

            # When done training, get the prediction
            prediction = sess.run([tf.gather(getattr(self, "models").get(var),
                                             mhf.match_indices(larger=self.resolution,
                                                               smaller=getattr(self, "data_shape")[0]))])

        return prediction

    def predict(self, vars=None):
        data = {}
        if vars is not None:
            for var in vars:
                data[var] = getattr(self, "predictions").get(var)
        else:
            for var in getattr(self, "x_vars") + [getattr(self, "y_var")]:
                data[var] = getattr(self, "predictions").get(var)

        return data

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
        coefficients = {}

        for x_var in getattr(self, "x_vars"):
            if 1 == getattr(self, "dimension"):
                coefficients[x_var] = tf.Variable(initial_value=tf.random_normal([getattr(self, "spline_total") + 1],
                                                                                 mean=0.0, stddev=0.001),
                                                  dtype=tf.float32,
                                                  name=f"{x_var}_coefficients")
            else:
                coefficients[x_var] = tf.Variable(initial_value=tf.random_normal([getattr(self, "spline_total") + 1],
                                                                                 mean=0.0, stddev=0.001),
                                                  dtype=tf.float32,
                                                  name=f"{x_var}_coefficients")

        setattr(self, "coefficients", coefficients)

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

            model_splines = tensor_product(s1, s2)

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
        penalties = {}

        if self.penalize:
            for var in getattr(self, 'x_vars'):
                if 1 == getattr(self, "dimension"):
                    penalties[var] = mhf.quadratic_finite_difference(
                                getattr(self, "coefficients").get(var)[1:])
        else:
            for var in getattr(self, 'x_vars'):
                setattr(self, f"{var}_penalty", tf.constant(0., dtype=tf.float32))

        setattr(self, "penalties", penalties)

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

            objectives[var] = obj + getattr(self, "penalties").get(var)

        # If there is a Y variable, define its objective function.
        if isinstance(getattr(self, "y_var"), str):
            with getattr(self, "y_var") as y_var:
                obj = tf.reduce_sum(tf.square(tf.subtract(
                    tf.gather(getattr(self, 'models').get(y_var),
                              indices=mhf.match_indices(larger=self.resolution,
                                                        smaller=getattr(self, "data_shape")[0])),
                    getattr(self, 'placeholders').get(y_var))),
                    name=f"{y_var}_objective")

                objectives[y_var] = obj

        setattr(self, "objectives", objectives)

    def _make_optimizers(self):
        optimizers = {}
        for var in getattr(self, 'x_vars'):
            learning_rate = tf.train.exponential_decay(learning_rate=20., global_step=self.global_step,
                                                       decay_steps=100, decay_rate=0.90, staircase=True,
                                                       name=f"{var}_learning_rate")
            optimizer = tf.train.AdadeltaOptimizer(learning_rate=learning_rate, name=f"{var}_optimizer")
            optimizers[var] = optimizer

        setattr(self, "optimizers", optimizers)

    def add_indicator(self, **kwargs):
        pass

