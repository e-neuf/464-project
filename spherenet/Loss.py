import torch
from SDF import determine_sphere_sdf, determine_cone_sdf
import numpy as np
import torch.nn.functional as F

########################
## Cone Loss Functions##
########################

def cone_overlap_loss(cone_params):
    center_distances = torch.cdist(cone_params[:, :3], cone_params[:, :3])
    radii_sums = cone_params[:, 3:4] + cone_params[:, 3:4].T
    overlap = torch.relu(radii_sums - center_distances)
    return torch.mean(overlap)

def calculate_graded_outside_loss_cone_sdf(cone_params, sdf_points, sdf_values, penalty_scale=2.0):
    cone_centers = cone_params[:, :3] 
    cone_radii = cone_params[:, 3]    
    cone_heights = cone_params[:, 4]  

    # Treat each cone as a sphere with radius = max(radius, height)
    sphere_radii = torch.max(cone_radii, cone_heights)  

    # SDF for cone centers
    center_sdf_distances = torch.cdist(cone_centers, sdf_points)  
    closest_sdf_indices = torch.argmin(center_sdf_distances, dim=1)  
    closest_sdf_values = sdf_values[closest_sdf_indices]  

    # Penalize spheres with positive SDF values (outside model)
    outside_penalty = torch.relu(closest_sdf_values + sphere_radii)  

    # quadratic penatly
    graded_penalty = penalty_scale * torch.mean(outside_penalty**2)

    return graded_penalty

def cone_overlap_loss(cone_params, penalty_scale=1.0):
    cone_centers = cone_params[:, :3]  
    cone_radii = cone_params[:, 3]    
    cone_heights = cone_params[:, 4]  

    # Treat each cone as a sphere with radius = max(radius, height)
    sphere_radii = torch.max(cone_radii, cone_heights)  

    # pairwise distances between cone centers
    center_distances = torch.cdist(cone_centers, cone_centers) 

    # pairwise sums of sphere radii
    radii_sums = sphere_radii[:, None] + sphere_radii[None, :] 

    overlap = torch.relu(radii_sums - center_distances) 
    overlap.fill_diagonal_(0.0)

    overlap_loss = penalty_scale * torch.sum(overlap ** 2)

    return overlap_loss

def calculate_inside_cone_coverage_loss(sdf_points, sdf_values, cone_params):
    # inside or surface points (SDF <= 0)
    inside_mask = sdf_values <= 0
    inside_points = sdf_points[inside_mask]
    inside_sdf_values = sdf_values[inside_mask]

    if inside_points.shape[0] == 0:  
        return torch.tensor(0.0, device=sdf_points.device)

    cone_sdf = determine_cone_sdf(inside_points, cone_params)  
    min_sdf, _ = torch.min(cone_sdf, dim=1)  

    # cone SDF > ground truth SDF, penalize
    uncovered_loss = torch.mean(torch.relu(min_sdf - inside_sdf_values)) 
    over_coverage_loss = torch.mean(torch.relu(-min_sdf - 1.0))
    total_loss = uncovered_loss + 0.1 * over_coverage_loss

    return total_loss


##########################
## Sphere Loss Functions##
##########################


def calculate_huber_loss(predictions, targets, delta=1.0):
    error = predictions - targets
    abs_error = torch.abs(error)
    
    quadratic = torch.where(abs_error <= delta, 0.5 * error ** 2, torch.zeros_like(error))
    linear = torch.where(abs_error > delta, delta * abs_error - 0.5 * delta ** 2, torch.zeros_like(error))
    
    loss = quadratic + linear
    return torch.mean(loss)

def calculate_overlap_loss(sphere_params):
    centers = sphere_params[:, :3]
    radii = sphere_params[:, 3]
    
    # Calculate pairwise distances between sphere centers
    dist_matrix = torch.cdist(centers, centers)
    
    # Calculate pairwise sum of radii
    radii_matrix = radii[:, None] + radii[None, :]
    
    overlap_matrix = torch.relu(radii_matrix - dist_matrix)
    # Sum of squared overlaps 
    overlap_loss = torch.sum(overlap_matrix ** 2) - torch.sum(torch.diag(overlap_matrix ** 2))
    
    return overlap_loss

def calculate_inside_coverage_loss(sdf_points, sdf_values, sphere_params):
    # Get inside points (SDF values < 0)
    inside_mask = sdf_values < 0
    inside_points = sdf_points[inside_mask]
    
    if inside_points.shape[0] == 0:  #
        return torch.tensor(0.0, device=sdf_points.device)

    # Calculate SDF for these points wrt spheres
    sphere_sdf = determine_sphere_sdf(inside_points, sphere_params)
    
    min_sdf, _ = torch.min(sphere_sdf, dim=1)
    uncovered_loss = torch.mean(torch.relu(min_sdf))  
    
    return uncovered_loss

def calculate_graded_outside_loss(sphere_params, voxel_bounds, buffer=2.0, penalty_scale=2.0):
    sphere_centers = sphere_params[:, :3]
    sphere_radii = sphere_params[:, 3]
    
    # Voxel bounds
    (xmin, ymin, zmin), (xmax, ymax, zmax) = voxel_bounds

    # distances outside the bounds
    outside_xmin = torch.clamp(xmin - (sphere_centers[:, 0] - sphere_radii) - buffer, min=0)
    outside_ymin = torch.clamp(ymin - (sphere_centers[:, 1] - sphere_radii) - buffer, min=0)
    outside_zmin = torch.clamp(zmin - (sphere_centers[:, 2] - sphere_radii) - buffer, min=0)

    outside_xmax = torch.clamp((sphere_centers[:, 0] + sphere_radii) - xmax - buffer, min=0)
    outside_ymax = torch.clamp((sphere_centers[:, 1] + sphere_radii) - ymax - buffer, min=0)
    outside_zmax = torch.clamp((sphere_centers[:, 2] + sphere_radii) - zmax - buffer, min=0)

    # quadratic penalty
    penalty_x = outside_xmin ** 2 + outside_xmax ** 2
    penalty_y = outside_ymin ** 2 + outside_ymax ** 2
    penalty_z = outside_zmin ** 2 + outside_zmax ** 2

    outside_loss = penalty_scale * torch.mean(penalty_x + penalty_y + penalty_z)
    return outside_loss

def penalize_large_spheres(sphere_params):
    sphere_radii = sphere_params[:, 3]
    return torch.mean(sphere_radii ** 2)  

############################
## Cylinder Loss Functions##
############################

def penalize_large_cylinders(cylinder_params):
    cylinder_radii = cylinder_params[:,6]
    cylinder_height = cylinder_params[:,7]
    return torch.mean(cylinder_radii ** 2 + cylinder_height ** 2)
