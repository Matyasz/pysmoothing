import numpy as np
import scipy.ndimage


def create_splines(quantity: int, resolution: int=500, order: int=3, const_spline: bool=True):
    """
        Creates and returns a given number of splines as a numpy array.
        Uses De Boor's algorithm to recursively create the splines.
        Also offers an option to use two sets of splines to create 3-dimensional splines.

        :param quantity:      The number of splines.
        :param resolution:    The length of the spline vectors to be returned.
        :param order:         The order of the splines. Default is cubic splines.
        :param const_spline:  Whether or not to include a constant spline

    """
    domain = np.asarray(np.linspace(start=0.0, stop=1.0, num=resolution))

    # Need enough knots for the algorithm to generate the correct number of k^th order splines
    knots = np.asarray(np.linspace(start=domain[0],
                                   stop=domain[-1],
                                   num=(quantity + order + 1)))

    if not isinstance(quantity, int):
        raise TypeError("quantity must be an integer")

    if not isinstance(resolution, int):
        raise TypeError("resolution must be an integer")

    if not isinstance(order, int):
        raise TypeError("order must be an integer")

    # Now create the splines
    # Make the initial order-0 splines to run the algorithm on
    spline_0 = np.asarray([[1 if knots[i] <= domain[j] <= knots[i + 1] else 0 for j in range(resolution)]
                           for i in range(0, len(knots) - 1)])

    # The recursive algorithm to generate the higher order splines
    def recursive_de_boors(i, ord):
        if 0 == ord:
            return spline_0[i, :]
        else:
            coeff_one = np.array((domain - knots[i]) / (knots[i + ord] - knots[i]))
            coeff_two = np.array((knots[i + ord + 1] - domain) / (knots[i + ord + 1] - knots[i + 1]))

            return (coeff_one * recursive_de_boors(i, ord - 1)) + (coeff_two * recursive_de_boors(i + 1, ord - 1))

    # Now call the algorithm for each spline
    splines = []
    for i in range(0, quantity):
        splines.append(recursive_de_boors(i, order))
    splines = np.asarray(splines)

    # Now we "clip" the splines:
    #
    # For the smoothing model to work well at the endpoints, we need to clip the splines so that the
    # ones at the ends are only partial splines. Otherwise, the endpoint measurements would always be 0.

    # Get the indices to clip
    left_clip = np.argmax(splines[0]) + abs(splines[0, np.argmax(splines[0]):np.argmax(splines[2])] -
                                            splines[2, np.argmax(splines[0]):np.argmax(splines[2])]).argmin()
    right_cip = np.argmax(splines[-3]) + abs(splines[-3, np.argmax(splines[-3]):np.argmax(splines[-1])] -
                                             splines[-1, np.argmax(splines[-3]):np.argmax(splines[-1])]).argmin()

    splines = splines[:, left_clip:right_cip + 1]

    # Now we need to scale the width of the splines to match the user requested resolution
    scaled_clipped_splines = []
    for s in splines:
        scaled_clipped_splines.append(scipy.ndimage.zoom(s, resolution/len(s), order=3))
    del splines

    if const_spline:
        const = np.array([1.0 for _ in range(len(scaled_clipped_splines[0]))])
        scaled_clipped_splines = np.insert(scaled_clipped_splines, 0, const, axis=0)

    return scaled_clipped_splines


def tensor_product(splines_a, splines_b):
    splines = [np.outer(splines_a[0], splines_b[0])]

    for a in splines_a[1:]:
        for b in splines_b[1:]:
            splines.append(np.outer(a, b))
    return np.asarray(splines, dtype=np.float32)
