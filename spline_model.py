import model_helper_functions as mhf
from splines import create_splines

from numpy.linalg import norm
import tensorflow as tf


class SplineModel:
    def __init__(self, covariates, spline_count, x_vars, y_var: str=None,
                 penalize: bool=True, resolution: int=500, seed: int=0,
                 intra_op_parallelism_threads: int=0, inter_op_parallelism_threads: int=0):
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

        self.intra_op_parallelism_threads = intra_op_parallelism_threads
        self.inter_op_parallelism_threads = inter_op_parallelism_threads

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
        data = {getattr(self, "placeholders").get(var): data}

        prediction = self._fit(var, data, tolerance, print_freq)

        prediction = mhf.undo_normalize(prediction[0], mean, stdev)
        return prediction

    def fit_multi(self, fit_vars: list, raw_data: dict, tolerance: float=0.01, print_freq: int=500):
        """
            Used to fit all variables, and save a dictionary to the object which stores the fitted values.
        """
        predictions = {}

        # When fitting the Y Variable, add placeholders for all other variables to the feed dict
        data = {getattr(self, "placeholders").get(var): raw_data.get(var)
                for var in getattr(self, 'x_vars') + [getattr(self, 'y_var')]}

        for var in fit_vars:
            predictions[var] = self._fit(var, data, tolerance, print_freq)

        return predictions

    def _fit(self, var, data, tol, pr_freq):
        """
            Holds the logic for fitting a model to a variable.
        """
        gradients = getattr(self, "optimizers").get(var).compute_gradients(getattr(self, "objectives").get(var))
        minimizer = getattr(self, "optimizers").get(var).apply_gradients(gradients, global_step=self.global_step)

        counter = 0

        config = tf.ConfigProto(intra_op_parallelism_threads=self.intra_op_parallelism_threads,
                                inter_op_parallelism_threads=self.inter_op_parallelism_threads,
                                allow_soft_placement=True)

        print(f"Fitting {var}")
        with tf.Session(config=config) as sess:
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
            prediction = sess.run([tf.gather_nd(getattr(self, "models").get(var),
                                                mhf.match_indices(larger=self.resolution,
                                                                  smaller=getattr(self, "data_shape")))])

        return prediction

    def _make_placeholders(self):
        """
            Make the placeholder objects for the data that will be passed to the fit function by the user.
        """
        placeholders = {}

        # Make the placeholders for the penalty values
        for var in getattr(self, 'x_vars'):
            for cov in getattr(self, 'covariates'):
                ph_name = f"{var}_{cov}_penalty"
                placeholders[ph_name] = tf.placeholder(dtype=tf.float32,
                                                       shape=[1], name=ph_name)

        # Make the placeholders for the data to be fitted
        for var in getattr(self, 'x_vars') + [getattr(self, 'y_var')]:
            ph_name = f"{var}_placeholder"
            placeholders[var] = tf.placeholder(dtype=tf.float32,
                                               shape=getattr(self, "data_shape"), name=ph_name)

        setattr(self, "placeholders", placeholders)

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

            model_splines = tf.transpose(tensor_product(s1, s2))

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
                    penalties[var] = 0.
        else:
            for var in getattr(self, 'x_vars'):
                setattr(self, f"{var}_penalty", tf.constant(0., dtype=tf.float32))

        setattr(self, "penalties", penalties)

    def _make_objective_functions(self):
        objectives = {}

        # Define the objective functions for all X variable models.
        # Often, the splines will have far more input values than the actual data, so we need to make
        # sure that we grab the correct number of evenly spaced spline outputs for this calculation.
        indices = mhf.match_indices(larger=self.resolution, smaller=getattr(self, "data_shape"))
        data_indices = mhf.match_indices(larger=getattr(self, "data_shape"), smaller=getattr(self, "data_shape"))

        for var in getattr(self, 'x_vars'):
            model = tf.gather_nd(getattr(self, "models").get(var), indices=indices)
            data = tf.gather_nd(getattr(self, 'placeholders').get(var), indices=data_indices)

            # if 2 == getattr(self, "dimension"):
            #     # The multidimensional gather function returns a rank 1 tensor,
            #     # so we need to flatten the data to compare it to the  model
            #     model = tf.reshape(model, [getattr(self, "data_shape")[0], getattr(self, "data_shape")[1]])

            obj = tf.reduce_sum(tf.square(tf.subtract(model, data)),
                                name=f"{var}_objective")

            objectives[var] = obj + getattr(self, "penalties").get(var)

        # If there is a Y variable, define its objective function.
        if isinstance(getattr(self, "y_var"), str):
            with getattr(self, "y_var") as y_var:
                obj = tf.reduce_sum(tf.square(tf.subtract(
                    tf.gather(getattr(self, 'models').get(y_var),
                              indices=mhf.match_indices(larger=self.resolution,
                                                        smaller=getattr(self, "data_shape"))),
                    getattr(self, 'placeholders').get(y_var))),
                    name=f"{y_var}_objective")

                objectives[y_var] = obj

        setattr(self, "objectives", objectives)

    def _make_optimizers(self):
        optimizers = {}

        for var in getattr(self, 'x_vars'):
            learning_rate = tf.train.exponential_decay(learning_rate=20., global_step=self.global_step,
                                                       decay_steps=100, decay_rate=0.9, staircase=True,
                                                       name=f"{var}_learning_rate")
            optimizer = tf.train.AdadeltaOptimizer(learning_rate=learning_rate, name=f"{var}_optimizer")
            optimizers[var] = optimizer

        setattr(self, "optimizers", optimizers)

    def add_indicator(self, **kwargs):
        pass

