# smoothing
A package for smoothing 1-or-2 dimensional data using penalized basis splines with a penalty on the quadratic finite difference of the spline coefficients (as outlined in the paper Flexible smoothing􏰆 with B􏰀splines and penalties by Paul H. C. Eilers and and Brian D. Marx [https://pdfs.semanticscholar.org/5e3d/4cf7824be321af95ac098595957d8a87bf68.pdf]).

Various constraints allow for smoothing multiple datasets simultaneously. For example, one model can be forces to be the sum of two other models.

Any number of covariates can also be added to your model(s), and constraints can be applied to their coefficients (forced to be positive, negative, bounded between two values, etc.)
