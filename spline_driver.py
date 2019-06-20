import splines
import matplotlib.pyplot as plt

spline_count = 7
spl = splines.create_splines(spline_count, resolution=500)

fig = plt.figure()
for s in spl:
    plt.plot(s)

plt.show()