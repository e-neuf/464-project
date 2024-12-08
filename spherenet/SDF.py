import torch
import torch.nn.functional as F

def determine_sphere_sdf(query_points, sphere_params):
    sphere_centers = sphere_params[:,:3]
    sphere_radii = sphere_params[:,3]
    vectors_points_to_centers = query_points[:, None, :] - sphere_centers[None, :, :]
    distance_points_to_centers = torch.norm(vectors_points_to_centers, dim=-1)
    sphere_sdf = distance_points_to_centers - sphere_radii
    return sphere_sdf

def determine_cone_sdf(query_points, cone_params):

    if cone_params.dim() == 2:  # Shape: [num_cones, 8]
        cone_params = cone_params.unsqueeze(0)  
        
    cone_centers = cone_params[:, :, :3]
    cone_radii = cone_params[:, :, 3]
    cone_heights = cone_params[:, :, 4]
    cone_orientations = cone_params[:, :, 5:8]
    cone_orientations = F.normalize(cone_orientations, dim=-1)  # Ensure normalized orientations
    vectors_points_to_centers = query_points[:, None, :] - cone_centers[None, :, :, :]
    distance_points_to_centers = torch.norm(vectors_points_to_centers[:, :, :, :2], dim=-1)
    cone_sdf = torch.max(
        distance_points_to_centers - cone_radii * (1 - vectors_points_to_centers[:, :, :, 2] / cone_heights),
        torch.abs(vectors_points_to_centers[:, :, :, 2]) - cone_heights
    )
    return cone_sdf
