import spline_model
# import model_helper_functions as mh

import matplotlib.pyplot as plt
import numpy as np

years = np.array([y for y in range(1990, 2020)])
covariates = {'year': years}

exp = np.array([2 * np.log(y - min(years) + 1) + np.random.normal(0, 1) for y in years])


model = spline_model.SplineModel(covariates=covariates, spline_count=10, x_vars='exp')
model.make()

model.fit_one(x_var='exp', raw_data=exp)
fit = model.predict(vars=['exp'])['exp']


fig = plt.figure()
plt.scatter(years, exp)
plt.plot(years, np.array(fit).reshape((30,)))
plt.show()
