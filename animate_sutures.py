import matplotlib.pyplot as plt
import numpy as np
from matplotlib.animation import FuncAnimation
from MaxSuturePlacer import SuturePlacer

# example wound curve
sampled_points = [(x, 50 + 4 * np.sin(x / 8) * np.exp(-x / 150)) for x in range(0, 80)]

wound_width = 10
mm_per_pixel = 1

fig, ax = plt.subplots(figsize=(8, 5))
pts = np.array(sampled_points)
placer = SuturePlacer(wound_width, mm_per_pixel, sampled_points)

# compute high curv points
high_curv_idx = placer.compute_curvature_points(sampled_points)
high_curv_pts = pts[high_curv_idx]

interp_pts = placer.segment_along_curve(sampled_points, high_curv_idx)
interp_pts = np.array([pt for pt in interp_pts if tuple(pt) not in map(tuple, high_curv_pts)])

# define endpoints (also high curv points)
endpoints = np.array([pts[0], pts[-1]])
high_curv_pts = np.vstack([endpoints, high_curv_pts])  # ensure endpoints are in red set

# combine all points
full_pts = np.vstack([high_curv_pts, interp_pts]) if len(interp_pts) > 0 else high_curv_pts

# plot spline
ax.plot(pts[:, 0], pts[:, 1], 'k-', lw=1, label='Wound curve')

scatter_red = ax.scatter([], [], c='red', s=80, label='High curvature (incl. endpoints)')
scatter_green = ax.scatter([], [], c='green', s=50, label='Intermediate sutures')
scatter_blue = ax.scatter([], [], c='blue', s=50, label='Full suture plan')

ax.set_title("Suture Placement Animation")
ax.legend()
ax.set_xlim(min(pts[:, 0]) - 5, max(pts[:, 0]) + 5)
ax.set_ylim(min(pts[:, 1]) - 5, max(pts[:, 1]) + 5)

# animation update
total_frames = len(high_curv_pts) + len(interp_pts) + len(full_pts)

def update(frame):
    scatter_red.set_offsets(np.empty((0, 2)))
    scatter_green.set_offsets(np.empty((0, 2)))
    scatter_blue.set_offsets(np.empty((0, 2)))

    if frame < len(high_curv_pts):
        # high curv points (red)
        scatter_red.set_offsets(high_curv_pts[:frame + 1])
    elif frame < len(high_curv_pts) + len(interp_pts):
        # in btwn points (green)
        scatter_red.set_offsets(high_curv_pts)
        scatter_green.set_offsets(interp_pts[:frame - len(high_curv_pts) + 1])
    else:
        # full plan
        scatter_red.set_offsets(high_curv_pts)
        scatter_green.set_offsets(interp_pts)
        scatter_blue.set_offsets(full_pts[:frame - len(high_curv_pts) - len(interp_pts) + 1])

    return scatter_red, scatter_green, scatter_blue

ani = FuncAnimation(
    fig,
    update,
    frames=total_frames,
    interval=700,
    blit=True,
    repeat=True
)

plt.tight_layout()
plt.show()