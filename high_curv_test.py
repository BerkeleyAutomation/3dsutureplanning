import matplotlib.pyplot as plt
import numpy as np
from SuturePlacer import SuturePlacer

# define multiple sampled splines
splines = {
    "Sine Wave": [(50 + i*5, 100 + 20*np.sin(i/5)) for i in range(30)],
    "Cosine Wave": [(50 + i*5, 120 + 15*np.cos(i/6)) for i in range(30)],
    "Sine Variation": [(x, 50 + 20 * np.sin(x/5) * np.exp(-x/50)) for x in range(0, 60)],
    #"Multiple Dips": [(x, 100 + 1.5 * np.sin(x * 0.3)) for x in range(0, 50)],
    "Letter C": [(100 + 50*np.cos(theta), 100 + 50*np.sin(theta)) for theta in np.linspace(np.pi/2, 3*np.pi/2, 40)], 
    # "Sharp Corner": [(50, 100), (100, 100), (100, 150), (150, 150)]
}

# intialize parameters
wound_width = 10
mm_per_pixel = 1

fig, axes = plt.subplots(2, 2, figsize=(15, 12))
axes = axes.flatten()

for ax, (name, sampled_spline_pts) in zip(axes, splines.items()):
    placer = SuturePlacer(wound_width, mm_per_pixel, sampled_spline_pts)
    
    # compute high-curvature points and interpolated suture points
    high_curv_idx = placer.compute_curvature_points(sampled_spline_pts)
    suture_pts = placer.segment_along_curve(sampled_spline_pts, high_curv_idx)
    
    # convert to numpy arrays for plotting
    spline_pts_arr = np.array(sampled_spline_pts)
    suture_pts_arr = np.array(suture_pts)
    high_curv_pts_arr = spline_pts_arr[high_curv_idx]
    
    # plot 
    ax.plot(spline_pts_arr[:,0], spline_pts_arr[:,1], 'b-', label='Spline')
    ax.scatter(high_curv_pts_arr[:,0], high_curv_pts_arr[:,1], c='red', s=100, label='High Curvature')
    ax.scatter(suture_pts_arr[:,0], suture_pts_arr[:,1], c='green', s=50, label='Suture Points')
    ax.set_title(name)
    ax.legend()

plt.tight_layout()
plt.show()
