import torch
import torch.nn as nn
from SDF import determine_cone_sdf, determine_sphere_sdf


# class MetaNet(nn.Module):
#     def __init__(self, num_spheres=512, num_cones=32):
#         super(MetaNet, self).__init__()
#         self.sphere_weight = nn.Parameter(torch.tensor(0.5))  # Initialized to 0.5
#         self.cone_weight = nn.Parameter(torch.tensor(0.5))

#     def forward(self, sphere_params, cone_params, query_points):
#         """
#         Combine spheres and cones into a unified SDF representation.
#         """
#         # Compute individual SDFs
#         sphere_sdf = determine_sphere_sdf(query_points, sphere_params)
#         cone_sdf = determine_cone_sdf(query_points, cone_params)

#         # Blend SDFs using learnable weights
#         combined_sdf = self.sphere_weight * sphere_sdf + self.cone_weight * cone_sdf

#         # Normalize weights (optional)
#         combined_sdf /= (self.sphere_weight + self.cone_weight)

#         return combined_sdf, sphere_params, cone_params

class MetaNet(nn.Module):
    def __init__(self, num_spheres=512, num_cones=32):
        super(MetaNet, self).__init__()
        self.sphere_weights = nn.Parameter(torch.ones(num_spheres))  # Learnable weights for spheres
        self.cone_weights = nn.Parameter(torch.ones(num_cones))      # Learnable weights for cones

    def forward(self, sphere_params, cone_params, query_points):
        """
        Combine spheres and cones into a unified SDF representation.
        """
        # Compute individual SDFs
        sphere_sdf = determine_sphere_sdf(query_points, sphere_params)  # Shape: [num_points, num_spheres]
        cone_sdf = determine_cone_sdf(query_points, cone_params)        # Shape: [num_points, num_cones]

        # Apply weights to the SDFs
        weighted_sphere_sdf = self.sphere_weights * sphere_sdf          # Shape: [num_points, num_spheres]
        weighted_cone_sdf = self.cone_weights * cone_sdf                # Shape: [num_points, num_cones]

        # Combine SDFs using soft minimum
        combined_sdf = torch.min(weighted_sphere_sdf, dim=1).values + \
                       torch.min(weighted_cone_sdf, dim=1).values       # Shape: [num_points]

        return combined_sdf, self.sphere_weights, self.cone_weights
