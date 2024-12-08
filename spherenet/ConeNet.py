import torch
import torch.nn as nn
import numpy as np
import trimesh
from dgcnn import DGCNNFeat
from Decoder import Decoder
from SDF import determine_cone_sdf

class ConeNet(nn.Module):
    def __init__(self, num_cones=32):  # Adjusted num_cones to 256
        super(ConeNet, self).__init__()
        self.num_cones = num_cones
        self.encoder = DGCNNFeat(global_feat=True)
        self.decoder = Decoder(in_ch=256, out_ch=num_cones * 8)  # 8 parameters: center (3), radius (1), height (1), orientation (3)
        self.projection = nn.Linear(256, num_cones*8)  # Projection layer for initial cone centers

    def forward(self, voxel_data, query_points, initial_centers):
        # Pass the voxel data through the encoder
        features = self.encoder(voxel_data)
        features = features.view(features.size(0), -1)  # Flatten the features

        # Decode the features into cone parameters
        cone_params = self.decoder(features)

        cone_params = torch.sigmoid(cone_params.view(-1, self.num_cones, 8))

        # cone_params = self.projection(features)
        # cone_params = cone_params.view(features.size(0), self.num_cones, 8)  # Reshape to [batch_size, num_cones, 8]

        # cone_params[:, :, 3:5] = torch.relu(cone_params[:, :, 3:5])  # Radius and height


        cone_adder = torch.tensor([-0.5, -0.5, -0.5, 0.01, 0.01, -1.0, -1.0, -1.0]).to(cone_params.device)
        cone_multiplier = torch.tensor([1.0, 1.0, 1.0, 0.03, 0.03, 2.0, 2.0, 2.0]).to(cone_params.device)
        cone_params = cone_params * cone_multiplier + cone_adder

        # Use the initial centers for the first training iteration
        if self.training:
            updated_params = torch.cat((initial_centers.unsqueeze(0), cone_params[:, :, 3:]), dim=-1)
            cone_params = updated_params

            # cone_params[:, :, :3] = initial_centers

        cone_sdf = determine_cone_sdf(query_points, cone_params)

        return cone_sdf, cone_params

def compute_pca(data):
    # Center the data
    data_mean = np.mean(data, axis=0)
    centered_data = data - data_mean
    
    # Compute the covariance matrix
    covariance_matrix = np.cov(centered_data, rowvar=False)
    
    # Compute the eigenvalues and eigenvectors
    eigenvalues, eigenvectors = np.linalg.eigh(covariance_matrix)
    
    # Get the principal component (eigenvector with the largest eigenvalue)
    principal_component = eigenvectors[:, np.argmax(eigenvalues)]
    
    return principal_component

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

            # Ensure radius and height are scalar values
            radius = float(radius)
            height = float(height)

            # Extract points within the region of the cone
            mask = np.linalg.norm(points.cpu().detach().numpy() - center, axis=1) < radius
            region_points = points[mask].cpu().detach().numpy()
            
            # if len(region_points) > 0:
            #     # Compute the principal direction using PCA
            #     orientation = compute_pca(region_points)
            # else:
            #     # Default orientation along the z-axis if no points are found
            #     orientation = np.array([0, 0, 1])
            
            # Default orientation along the z-axis if no points are found
            # orientation = np.array([0, 0, 1])

            # Normalize the orientation vector
            orientation = orientation / np.linalg.norm(orientation)

            # Create the cone
            cone = trimesh.creation.cone(radius=radius, height=height)

            # Compute the rotation matrix to align the cone with the orientation vector
            z_axis = np.array([0, 0, 1])
            rotation_axis = np.cross(z_axis, orientation)
            if np.linalg.norm(rotation_axis) < 1e-6:
                rotation_matrix = np.eye(4)
            else:
                rotation_angle = np.arccos(np.dot(z_axis, orientation))
                rotation_matrix = trimesh.transformations.rotation_matrix(rotation_angle, rotation_axis)

            # Apply the rotation and translation to the cone
            cone.apply_transform(rotation_matrix)
            cone.apply_translation(center)
            scene.add_geometry(cone)

    inside_points = points[values < 0].cpu().detach().numpy()
    outside_points = points[values > 0].cpu().detach().numpy()
    if len(inside_points) > 0:
        inside_points = trimesh.points.PointCloud(inside_points)
        outside_points = trimesh.points.PointCloud(outside_points)
        inside_points.colors = np.array([[0, 0, 255, 255]] * len(inside_points.vertices))  # Blue color for inside points
        outside_points.colors = np.array([[0, 255, 255, 255]] * len(outside_points.vertices))  # Blue color for inside points
        scene.add_geometry([inside_points])

    if save_path is not None:
        scene.export(save_path)
    scene.show()
    # return voxel_data
