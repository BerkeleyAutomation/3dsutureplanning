import matplotlib.pyplot as plt
import numpy as np
from scipy.interpolate import splprep, splev

# Example spline: sine variation
sampled_points = [(x, 50 + 1.5 * np.sin(x / 10) * np.exp(-x / 100)) for x in range(0, 60)]
pts_arr = np.array(sampled_points)
x, y = pts_arr[:, 0], pts_arr[:, 1]

# Fit B-spline
tck, u = splprep([x, y], s=0, k=3)

# Dense evaluation for plotting
u_dense = np.linspace(0, 1, 500)
spline_x, spline_y = splev(u_dense, tck)

# Extract internal knots (remove repeated start/end knots)
k = 3
internal_knots = tck[0][k:-k]
knot_pts = np.array([splev(u_val, tck) for u_val in internal_knots])

# Plot spline and knots
plt.figure(figsize=(8, 6))
plt.plot(spline_x, spline_y, 'k-', lw=1, label='Spline')
if len(knot_pts) > 0:
    plt.scatter(knot_pts[:, 0], knot_pts[:, 1], c='purple', s=80, marker='x', label='Internal knots')

plt.title("B-spline Internal Knots")
plt.xlabel("X")
plt.ylabel("Y")
plt.legend()
plt.tight_layout()
plt.show()