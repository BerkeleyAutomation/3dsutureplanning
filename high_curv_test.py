import matplotlib.pyplot as plt
import numpy as np
from SuturePlacer import SuturePlacer

splines = {
    # very gentle curves
    "Sine Variation": [(x, 50 + 1.5 * np.sin(x / 10) * np.exp(-x / 100)) for x in range(0, 60)],
    "Multiple Dips": [(x, 50 + 0.5 * np.sin(x / 8)) for x in range(0, 50)],
}

wound_width = 10
mm_per_pixel = 1

fig, axes = plt.subplots(2, 2, figsize=(15, 12))
axes = axes.flatten()

for ax, (name, sampled_points) in zip(axes, splines.items()):
    # match your SuturePlacer signature: (wound_width, mm_per_pixel, sampled_spline_pts)
    placer = SuturePlacer(wound_width, mm_per_pixel, sampled_points)

    # compute high-curvature indices and suture points
    high_curv_idx = placer.compute_curvature_points(sampled_points)
    suture_pts = placer.segment_along_curve(sampled_points, high_curv_idx)

    spline_pts_arr = np.array(sampled_points)
    if len(high_curv_idx) > 0:
        high_curv_pts_arr = spline_pts_arr[high_curv_idx]
    else:
        high_curv_pts_arr = np.empty((0, 2))

    suture_pts_arr = np.array(suture_pts) if len(suture_pts) > 0 else np.empty((0, 2))

    ax.plot(spline_pts_arr[:, 0], spline_pts_arr[:, 1], 'b-', label='Sampled pts')
    if high_curv_pts_arr.size:
        ax.scatter(high_curv_pts_arr[:, 0], high_curv_pts_arr[:, 1], c='red', s=80, label='High curvature (indices)')
    if suture_pts_arr.size:
        ax.scatter(suture_pts_arr[:, 0], suture_pts_arr[:, 1], c='green', s=50, label='Suture points')

    ax.set_title(name)
    ax.legend()

plt.tight_layout()
plt.show()
