import torch
import torch.nn as nn
from dgcnn import DGCNNFeat
from SDF import determine_sphere_sdf
from Decoder import Decoder
import numpy as np
import trimesh

def visualise_spheres(points, values, sphere_params, reference_model=None, save_path=None):
    sphere_params = sphere_params.cpu().detach().numpy()
    sphere_centers = sphere_params[..., :3]
    sphere_radii = np.abs(sphere_params[..., 3])
    scene = trimesh.Scene()

    # Calculate the centroid of the sphere cluster
    centroid = sphere_centers.mean(axis=0)

    for center, radius in zip(sphere_centers, sphere_radii):
        sphere = trimesh.creation.icosphere(radius=radius, subdivisions=2)
        sphere.apply_translation(center)
        scene.add_geometry(sphere)

    inside_points = points[values < 0]
    inside_points = trimesh.points.PointCloud(inside_points)
    inside_points.colors = [0, 0, 255, 255]  # Blue color for inside points
    scene.add_geometry([inside_points])
        
    if save_path is not None:
        scene.export(save_path)
    scene.show()



class SphereNet(nn.Module):
    def __init__(self, num_spheres=512):
        super(SphereNet, self).__init__()
        self.num_spheres = num_spheres
        self.encoder = DGCNNFeat(global_feat=True)
        self.decoder = Decoder()

        self.feature_mapper = nn.Linear(512, num_spheres * 4)  # 4 parameters: center (3), radius (1)

    def forward(self, voxel_data, query_points):
        # Pass the voxel data through the encoder
        features = self.encoder(voxel_data)

        print(f"SphereNet Input Size: {features.shape}")
        
        sphere_params = self.decoder(features)
        print(f"Decoder Output Shape: {sphere_params.shape}")  # Debug decoder output shape

        sphere_params = self.feature_mapper(sphere_params).view(self.num_spheres, 4)

        sphere_params = torch.sigmoid(sphere_params.view(-1, 4))
        print(f"Sphere Params Shape: {sphere_params.shape}")  # Debug reshaped sphere_params

        sphere_adder = torch.tensor([-0.5, -0.5, -0.5, 0.01]).to(sphere_params.device)
        sphere_multiplier = torch.tensor([1.0, 1.0, 1.0, 0.2]).to(sphere_params.device)
        sphere_params = sphere_params * sphere_multiplier + sphere_adder
        sphere_sdf = determine_sphere_sdf(query_points, sphere_params)
        return sphere_sdf, sphere_params

