import spline_model

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

years = np.array([y for y in range(1996, 2016 + 1)])
ages = np.array([a for a in range(0, 90, 5)])

# covariates = {'year': years}
# exp = [2 * np.log(y - min(years) + 1) + np.random.normal(0, 0.5) for y in years]

covariates = {'year': years, 'age': ages}
exp = np.array([[a*a + (y - 1996) + np.random.normal(0, 0.1) for a in ages] for y in years])

# print(exp)

model = spline_model.SplineModel(covariates=covariates, spline_count=10, x_vars='exp')

model.make()

fit = model.fit_one(var='exp', raw_data=exp, tolerance=0.1, print_freq=100)

data = pd.DataFrame({'raw': exp.flatten()})
data['fit'] = fit

data.to_csv('/Users/taylor/Documents/outputs/data.csv', index=False)
# fig = plt.figure()
# plt.scatter(years, exp)
# plt.plot(years, np.array(fit).reshape((len(years),)))
# plt.show()

# data = pd.read_csv('/Users/taylor/Documents/outputs/data.csv')
# fig = plt.figure()
# plt.plot([x for x in range(len(data['fit']))], data['fit'])
# plt.scatter([x for x in range(len(data['raw']))], data['raw'])
# plt.show()
