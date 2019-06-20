import spline_model
# import model_helper_functions as mh

import matplotlib.pyplot as plt
import numpy as np

years = [y for y in range(1990, 2020)]
covariates = {'year': years}

exp = [2 * np.log(y - min(years)) + np.random.normal(0, 1) for y in years]

fig = plt.figure()
plt.scatter(years, exp)
plt.show()

# model = spline_model.SplineModel(covariates=covariates, spline_count=10, x_vars='exp')
# model.make()

# print(mh.match_indices(500, 20))
