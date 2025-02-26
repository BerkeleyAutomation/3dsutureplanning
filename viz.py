import cv2
import viser
import torch
import numpy as np
from scipy.spatial import cKDTree

DEVICE = 'cuda'

class VisualizePointcloudNode:

    def __init__(self):
        self.chicken_number = 3
        self.server_ = viser.ViserServer()

        self.image_callback()
        
    def estimate_normal(self, kdtree, point_cloud, query_point, k=30):
        """Estimate normal at query_point using PCA on k nearest neighbors."""
        _, idxs = kdtree.query(query_point, k)
        neighbors = point_cloud[idxs]  # (k, 3)

        # Compute mean-centered data
        mean = neighbors.mean(axis=0)
        centered = neighbors - mean

        # Compute PCA (Eigen decomposition of covariance matrix)
        cov = np.cov(centered.T)
        _, _, Vt = np.linalg.svd(cov)

        # Normal is the smallest eigenvector
        normal = Vt[-1]
        return normal / np.linalg.norm(normal)

    def draw_sphere(self, center, normal, radius=0.001, num_points=500):
        """Generate a sphere around the center point, oriented by the normal."""
        phi = np.linspace(0, np.pi, num_points // 2)
        theta = np.linspace(0, 2 * np.pi, num_points)
        phi, theta = np.meshgrid(phi, theta)
        
        u = np.cross(normal, np.array([1, 0, 0]))
        if np.linalg.norm(u) < 1e-6:
            u = np.cross(normal, np.array([0, 1, 0]))
        u /= np.linalg.norm(u)
        v = np.cross(normal, u)
        
        x = center[0] + radius * (np.sin(phi) * np.cos(theta) * u[0] + np.sin(phi) * np.sin(theta) * v[0] + np.cos(phi) * normal[0])
        y = center[1] + radius * (np.sin(phi) * np.cos(theta) * u[1] + np.sin(phi) * np.sin(theta) * v[1] + np.cos(phi) * normal[1])
        z = center[2] + radius * (np.sin(phi) * np.cos(theta) * u[2] + np.sin(phi) * np.sin(theta) * v[2] + np.cos(phi) * normal[2])
        
        sphere_points = np.vstack((x.flatten(), y.flatten(), z.flatten())).T
        return sphere_points, np.array([x[-1, -1], y[-1, -1], z[-1, -1]])  # Return top of sphere as well

    def image_callback(self):
        self.left_img = cv2.imread(f'dan_chicken/left_exp_00{self.chicken_number}.png')
        self.right_img = cv2.imread(f'dan_chicken/right_exp_00{self.chicken_number}.png')
        point_cloud_data = np.load(f'dan_point_cloud_data/point_cloud{self.chicken_number}.npy')
        rgb_cloud_data = np.load(f'dan_point_cloud_data/rgb_cloud{self.chicken_number}.npy')

        kdtree = cKDTree(point_cloud_data)
        self.insertion = np.load(f'dan_insertion_extraction_pts/insertion_pts{self.chicken_number}.npy')
        self.extraction = np.load(f'dan_insertion_extraction_pts/extraction_pts{self.chicken_number}.npy')

        self.server_.add_point_cloud('mesh', points=point_cloud_data, colors=rgb_cloud_data, point_size=0.001, point_shape='sparkle')

        insertion_tops = []
        extraction_tops = []

        for i, pt in enumerate(self.insertion):
            normal = self.estimate_normal(kdtree, point_cloud_data, pt)
            sphere_pts, top_pt = self.draw_sphere(pt, normal)
            insertion_tops.append(top_pt)
            insertion_points_color = [[146, 243, 135] for i in range(len(sphere_pts))] # green
            self.server_.add_point_cloud(f'insertion_points{i}', points=sphere_pts, colors=insertion_points_color, point_size=0.00009, point_shape='sparkle')

        for i, pt in enumerate(self.extraction):
            normal = self.estimate_normal(kdtree, point_cloud_data, pt)
            sphere_pts, top_pt = self.draw_sphere(pt, normal)
            extraction_tops.append(top_pt)
            extraction_points_color = [[255, 71, 71] for i in range(len(sphere_pts))] # red
            self.server_.add_point_cloud(f'extraction_points{i}', points=sphere_pts, colors=extraction_points_color, point_size=0.00009, point_shape='sparkle')

        line_points = np.array([x for x in zip(insertion_tops, extraction_tops)])
        line_color = np.array([[[0, 0, 0], [0, 0, 0]] for i in range(len(line_points))])
        self.server_.add_line_segments("lines", points=line_points, colors=line_color, line_width=2)
        import pdb
        pdb.set_trace()
        exit()

if __name__ == '__main__':
    visualize_pointcloud_node = VisualizePointcloudNode()
