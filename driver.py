import spline_model
# import model_helper_functions as mh

import matplotlib.pyplot as plt
import numpy as np

years = np.array([y for y in range(1996, 2016 + 1)])
covariates = {'year': years}

exp = np.array([2 * np.log(y - min(years) + 1) + np.random.normal(0, 0.5) for y in years])


model = spline_model.SplineModel(covariates=covariates, spline_count=10, x_vars='exp')
model.make()

fit = model.fit_one(var='exp', raw_data=exp)

fig = plt.figure()
plt.scatter(years, exp)
plt.plot(years, np.array(fit).reshape((len(years),)))
plt.show()
