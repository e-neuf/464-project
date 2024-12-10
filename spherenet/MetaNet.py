import torch
import torch.nn as nn
from SDF import determine_cone_sdf, determine_sphere_sdf

class MetaNet(nn.Module):
    def __init__(self, num_spheres=512, num_cones=32):
        super(MetaNet, self).__init__()
        self.sphere_weights = nn.Parameter(torch.ones(num_spheres))  # Learnable weights for spheres
        self.cone_weights = nn.Parameter(torch.ones(num_cones))      # Learnable weights for cones

    def forward(self, sphere_params, cone_params, query_points):
        sphere_sdf = determine_sphere_sdf(query_points, sphere_params)  
        cone_sdf = determine_cone_sdf(query_points, cone_params)      

        # Apply weights to the SDFs
        weighted_sphere_sdf = self.sphere_weights * sphere_sdf        
        weighted_cone_sdf = self.cone_weights * cone_sdf               

        # Combine SDFs using soft minimum
        combined_sdf = torch.min(weighted_sphere_sdf, dim=1).values + \
                       torch.min(weighted_cone_sdf, dim=1).values       

        return combined_sdf, self.sphere_weights, self.cone_weights
