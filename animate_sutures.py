import matplotlib.pyplot as plt
import numpy as np
from matplotlib.animation import FuncAnimation
from SuturePlacer import SuturePlacer

# Example splines
splines = {
    "Sine Variation (1)": [(x, 50 + 1.5 * np.sin(x / 10) * np.exp(-x / 100)) for x in range(0, 60)],
    "Sine Variation (2)": [(x, 50 + 0.5 * np.sin(x / 8)) for x in range(0, 50)],
}

wound_width = 10
mm_per_pixel = 1

fig, axes = plt.subplots(1, len(splines), figsize=(10 * len(splines), 6))
if len(splines) == 1:
    axes = [axes]

for ax, (name, sampled_points) in zip(axes, splines.items()):
    placer = SuturePlacer(wound_width, mm_per_pixel, sampled_points)

    # Compute high-curvature indices
    high_curv_idx = placer.compute_curvature_points(sampled_points)
    high_curv_pts = np.array(sampled_points)[high_curv_idx]

    # Compute intermediate points along curve
    interp_pts = placer.segment_along_curve(sampled_points, high_curv_idx)
    interp_pts = np.array([pt for pt in interp_pts if tuple(pt) not in map(tuple, high_curv_pts)])

    # Full suture set
    full_pts = np.vstack([high_curv_pts, interp_pts])

    # Plot base spline
    spline_pts_arr = np.array(sampled_points)
    ax.plot(spline_pts_arr[:, 0], spline_pts_arr[:, 1], 'k-', lw=1, label='Spline')

    # Initialize scatter plots with empty Nx2 arrays
    scatter_red = ax.scatter(np.empty((0, 2)), np.empty((0, 2)), c='red', s=80, label='High-curvature')
    scatter_green = ax.scatter(np.empty((0, 2)), np.empty((0, 2)), c='green', s=50, label='Intermediate')
    scatter_blue = ax.scatter(np.empty((0, 2)), np.empty((0, 2)), c='blue', s=50, label='All sutures')
    ax.set_title(name)
    ax.legend()

    # Total frames for animation
    total_frames = len(high_curv_pts) + len(interp_pts) + len(full_pts)

    def update(frame):
        # Reset offsets each frame
        scatter_red.set_offsets(np.empty((0, 2)))
        scatter_green.set_offsets(np.empty((0, 2)))
        scatter_blue.set_offsets(np.empty((0, 2)))

        if frame < len(high_curv_pts):
            scatter_red.set_offsets(high_curv_pts[:frame + 1])
        elif frame < len(high_curv_pts) + len(interp_pts):
            scatter_red.set_offsets(high_curv_pts)
            scatter_green.set_offsets(interp_pts[:frame - len(high_curv_pts) + 1])
        else:
            scatter_red.set_offsets(high_curv_pts)
            scatter_green.set_offsets(interp_pts)
            scatter_blue.set_offsets(full_pts[:frame - len(high_curv_pts) - len(interp_pts) + 1])
        return scatter_red, scatter_green, scatter_blue

    ani = FuncAnimation(fig, update, frames=total_frames, interval=700, blit=True, repeat=True)

plt.tight_layout()
plt.show()
