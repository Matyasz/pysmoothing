import numpy as np
import scipy.ndimage


def create_splines(domain: (int, int), quantity: int, resolution: int=100, order: int=3):
    """
            Creates and returns a given number of splines as a numpy array.
            Uses De Boor's algorithm to recursively create the splines.
            Also offers an option to use two sets of splines to create 3-dimensional splines.

            :param domain:     The resolution desired for hte splines.
            :param quantity:   The number of splines.
            :param resolution: The length of the spline vectors to be returned.
            :param order:      The order of the splines. Default is cubic splines.

    """

    # First validate the inputs and
    if not isinstance(domain, tuple):
        raise TypeError("domain should be a tuple of length 2: Currently not a tuple")
    elif isinstance(domain, tuple) and 2 != len(domain):
        raise AttributeError(f"domain should be a tuple of length 2: Currently length is {len(domain)}")
    else:
        # Need enough knots for the algorithm to generate the correct number of k^th order splines
        knots = np.asarray(np.linspace(start=domain[0],  # - (0.3 * abs(domain[0])),
                                       stop=domain[1],  # + (0.301 * abs(domain[1])),
                                       num=(quantity + order + 1)))

        domain = np.asarray(np.linspace(start=domain[0], stop=domain[1], num=resolution))

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

    """
            Now we "clip" the splines:
    
            For the smoothing model to work well at the endpoints, we need to clip the splines so that the
            ones at the ends are only partial splines. Otherwise, the endpoint measurements would always be 0.
    """

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
    return scaled_clipped_splines


def tensor_product(spl_a, spl_b, include_const_layer=False):
    pass
