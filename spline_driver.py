import splines
import matplotlib.pyplot as plt

spline_count = 8
A = splines.create_splines(spline_count)
B = splines.create_splines(spline_count)

C = splines.tensor_product(A, B)

plt.imshow(C[36], cmap='hot', interpolation='nearest')
plt.show()
