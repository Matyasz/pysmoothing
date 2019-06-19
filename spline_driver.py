import splines
import matplotlib.pyplot as plt

spline_count = 7
spl = splines.create_splines(spline_count, resolution=500)

fig = plt.figure()
for i in range(spline_count):
    plt.plot(spl[i])

plt.show()