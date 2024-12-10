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
    """
    Visualize cones using their parameters and SDF data.

    Args:
        points (torch.Tensor): Points in the 3D space, shape [num_points, 3].
        values (torch.Tensor): SDF values for the points, shape [num_points].
        cone_params (torch.Tensor): Parameters for cones, shape [x, 8].
        save_path (str, optional): Path to save the visualization. Defaults to None.
    """
    # Ensure cone_params is of shape [x, 8]
    cone_params = cone_params.squeeze(0).cpu().detach().numpy() 
    cone_centers = cone_params[:, :3]
    cone_radii = np.abs(cone_params[:, 3])
    cone_heights = np.abs(cone_params[:, 4])
    cone_orientations = cone_params[:, 5:8]
    scene = trimesh.Scene()

    for i in range(cone_centers.shape[0]):
        center = cone_centers[i]
        radius = cone_radii[i]
        height = cone_heights[i]
        orientation = cone_orientations[i]

        radius = float(radius)
        height = float(height)

        # Extract points within the region of the cone
        mask = np.linalg.norm(points.cpu().detach().numpy() - center, axis=1) < radius
        region_points = points[mask].cpu().detach().numpy()

        # Normalize the orientation vector
        if np.linalg.norm(orientation) < 1e-6:
            orientation = np.array([0, 0, 1])  
        orientation = orientation / np.linalg.norm(orientation)

        cone = trimesh.creation.cone(radius=radius, height=height)

        # Compute the rotation matrix to align the cone with the orientation vector
        z_axis = np.array([0, 0, 1])
        rotation_axis = np.cross(z_axis, orientation)
        if np.linalg.norm(rotation_axis) < 1e-6:
            rotation_matrix = np.eye(4)  #identity matrix
        else:
            rotation_angle = np.arccos(np.dot(z_axis, orientation))
            rotation_matrix = trimesh.transformations.rotation_matrix(rotation_angle, rotation_axis)

        cone.apply_transform(rotation_matrix)
        cone.apply_translation(center)
        scene.add_geometry(cone)

    inside_points = points[values < 0].cpu().detach().numpy()
    if len(inside_points) > 0:
        inside_points = trimesh.points.PointCloud(inside_points)
        inside_points.colors = np.array([[0, 0, 255, 255]] * len(inside_points.vertices))  
        scene.add_geometry([inside_points])

    if save_path is not None:
        scene.export(save_path)
    scene.show()
