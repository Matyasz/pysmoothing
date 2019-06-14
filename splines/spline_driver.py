import splines
import matplotlib.pyplot as plt

spline_count = 10
spl = splines.create_splines((0, 100), spline_count, resolution=250)

for s in spl:
    print(s)

fig = plt.figure()
for i in range(spline_count):
    plt.plot(spl[i])

plt.show()