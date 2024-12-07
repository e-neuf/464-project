import torch
import torch.nn as nn
from dgcnn import DGCNNFeat
from SDF import determine_sphere_sdf
from Decoder import Decoder

class SphereNet(nn.Module):
    def __init__(self, num_spheres=512):
        super(SphereNet, self).__init__()
        self.num_spheres = num_spheres
        self.encoder = DGCNNFeat(global_feat=True)
        self.decoder = Decoder()

    def forward(self, voxel_data, query_points):
        # Pass the voxel data through the encoder
        features = self.encoder(voxel_data)
        sphere_params = self.decoder(features)
        sphere_params = torch.sigmoid(sphere_params.view(-1, 4))
        sphere_adder = torch.tensor([-0.5, -0.5, -0.5, 0.01]).to(sphere_params.device)
        sphere_multiplier = torch.tensor([1.0, 1.0, 1.0, 0.2]).to(sphere_params.device)
        sphere_params = sphere_params * sphere_multiplier + sphere_adder
        sphere_sdf = determine_sphere_sdf(query_points, sphere_params)
        return sphere_sdf, sphere_params

