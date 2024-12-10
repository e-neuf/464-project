import torch
import torch.nn as nn
import numpy as np
import trimesh
from dgcnn import DGCNNFeat
from Decoder import Decoder
from SDF import determine_cone_sdf
from sklearn.cluster import KMeans


class ConeNet(nn.Module):
    def __init__(self, num_cones=32): 
        super(ConeNet, self).__init__()
        self.num_cones = num_cones
        self.encoder = DGCNNFeat(global_feat=True)
        self.decoder = Decoder()
        self.feature_mapper = nn.Linear(512, num_cones * 8)  

    def forward(self, voxel_data, query_points):
        # Pass the voxel data through the encoder
        features = self.encoder(voxel_data)
        # Decode the features into cone parameters
        cone_params = self.decoder(features)

        cone_params = self.feature_mapper(cone_params).view(self.num_cones, 8)
        cone_params = torch.sigmoid(cone_params.view(-1, 8))

        cone_adder = torch.tensor([-0.5, -0.5, -0.5, 0.0, 0.0, -1.0, -1.0, -1.0]).to(cone_params.device)
        cone_multiplier = torch.tensor([1.0, 1.0, 1.0, 0.1, 0.1, 2.0, 2.0, 2.0]).to(cone_params.device)
        cone_params = cone_params * cone_multiplier + cone_adder

        cone_sdf = determine_cone_sdf(query_points, cone_params)

        return cone_sdf, cone_params

def visualize_cones(points, values, cone_params, save_path=None):
    cone_params = cone_params.cpu().detach().numpy()
    cone_centers = cone_params[..., :3]
    cone_radii = np.abs(cone_params[..., 3])
    cone_heights = np.abs(cone_params[..., 4])
    cone_orientations = cone_params[..., 5:8]
    scene = trimesh.Scene()

    for i in range(cone_centers.shape[0]):
        for j in range(cone_centers.shape[1]):
            center = cone_centers[i, j]
            radius = cone_radii[i, j]
            height = cone_heights[i, j]
            orientation = cone_orientations[i, j]

            # scalar values
            radius = float(radius)
            height = float(height)

            # Extract points within the region of the cone
            mask = np.linalg.norm(points.cpu().detach().numpy() - center, axis=1) < radius
            region_points = points[mask].cpu().detach().numpy()

            # Normalize the orientation vector
            orientation = orientation / np.linalg.norm(orientation)

            # Create the cone
            cone = trimesh.creation.cone(radius=radius, height=height)

            # rotation matrix with the orientation vector
            z_axis = np.array([0, 0, 1])
            rotation_axis = np.cross(z_axis, orientation)
            if np.linalg.norm(rotation_axis) < 1e-6:
                rotation_matrix = np.eye(4)
            else:
                rotation_angle = np.arccos(np.dot(z_axis, orientation))
                rotation_matrix = trimesh.transformations.rotation_matrix(rotation_angle, rotation_axis)

            cone.apply_transform(rotation_matrix)
            cone.apply_translation(center)
            scene.add_geometry(cone)

    inside_points = points[values < 0].cpu().detach().numpy()
    if len(inside_points) > 0:
        inside_points = trimesh.points.PointCloud(inside_points)
        inside_points.colors = np.array([[0, 0, 255, 255]] * len(inside_points.vertices))  # Blue color for inside points
        scene.add_geometry([inside_points])

    if save_path is not None:
        scene.export(save_path)
    scene.show()
