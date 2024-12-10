import torch
from SDF import determine_sphere_sdf, determine_cone_sdf
import numpy as np
import torch.nn.functional as F


# Cone Loss Functions

def cone_size_diversity_loss(cone_params):
    radii = cone_params[..., 3]
    heights = cone_params[..., 4]
    return -torch.std(radii) - torch.std(heights)  # Encourage diversity

def penalize_large_cones(cone_params):
    cone_radii = cone_params[:, 3]
    cone_heights = cone_params[:, 4]
    return torch.mean(cone_radii ** 2 + cone_heights ** 2)

def cone_overlap_loss(cone_params):
    center_distances = torch.cdist(cone_params[:, :3], cone_params[:, :3])
    radii_sums = cone_params[:, 3:4] + cone_params[:, 3:4].T
    overlap = torch.relu(radii_sums - center_distances)
    return torch.mean(overlap)

def calculate_graded_outside_loss_cone_sdf(cone_params, sdf_points, sdf_values, penalty_scale=2.0):
    """
    Penalize cones treated as spheres if they extend outside the model.

    Args:
        cone_params (torch.Tensor): Parameters for cones, shape [num_cones, 8].
                                     Each cone has parameters [center_x, center_y, center_z, radius, height, dir_x, dir_y, dir_z].
        sdf_points (torch.Tensor): Query points in 3D space, shape [num_points, 3].
        sdf_values (torch.Tensor): SDF values for the query points, shape [num_points].
        penalty_scale (float): Scale factor for the penalty.

    Returns:
        torch.Tensor: The total outside penalty (scalar).
    """
    # Extract cone parameters
    cone_centers = cone_params[:, :3]  # Shape: [num_cones, 3]
    cone_radii = cone_params[:, 3]    # Shape: [num_cones]
    cone_heights = cone_params[:, 4]  # Shape: [num_cones]

    # Treat each cone as a sphere with radius = max(radius, height)
    sphere_radii = torch.max(cone_radii, cone_heights)  # Shape: [num_cones]

    # Compute SDF values for cone centers
    center_sdf_distances = torch.cdist(cone_centers, sdf_points)  # Shape: [num_cones, num_sdf_points]
    closest_sdf_indices = torch.argmin(center_sdf_distances, dim=1)  # Shape: [num_cones]
    closest_sdf_values = sdf_values[closest_sdf_indices]  # Shape: [num_cones]

    # Penalize spheres whose centers have positive SDF values (outside the model)
    outside_penalty = torch.relu(closest_sdf_values + sphere_radii)  # Penalize based on radius

    # Apply graded penalty (e.g., quadratic)
    graded_penalty = penalty_scale * torch.mean(outside_penalty**2)

    return graded_penalty

def cone_overlap_loss(cone_params, penalty_scale=1.0):
    """
    Penalize overlap between cones treated as spheres.

    Args:
        cone_params (torch.Tensor): Parameters for cones, shape [num_cones, 8].
                                     Each cone has parameters [center_x, center_y, center_z, radius, height, dir_x, dir_y, dir_z].
        penalty_scale (float): Scale factor for the penalty.

    Returns:
        torch.Tensor: The total overlap penalty (scalar).
    """
    # Extract cone parameters
    cone_centers = cone_params[:, :3]  # Shape: [num_cones, 3]
    cone_radii = cone_params[:, 3]    # Shape: [num_cones]
    cone_heights = cone_params[:, 4]  # Shape: [num_cones]

    # Treat each cone as a sphere with radius = max(radius, height)
    sphere_radii = torch.max(cone_radii, cone_heights)  # Shape: [num_cones]

    # Compute pairwise distances between cone centers
    center_distances = torch.cdist(cone_centers, cone_centers)  # Shape: [num_cones, num_cones]

    # Compute pairwise sums of sphere radii
    radii_sums = sphere_radii[:, None] + sphere_radii[None, :]  # Shape: [num_cones, num_cones]

    # Calculate overlaps
    overlap = torch.relu(radii_sums - center_distances)  # Shape: [num_cones, num_cones]

    # Exclude self-overlaps (diagonal)
    overlap.fill_diagonal_(0.0)

    # Penalize overlaps (e.g., quadratic penalty)
    overlap_loss = penalty_scale * torch.sum(overlap ** 2)

    return overlap_loss

def calculate_inside_cone_coverage_loss(sdf_points, sdf_values, cone_params):
    """
    Penalize cones that fail to cover inside points of the voxel grid (SDF <= 0), encouraging tight and accurate coverage.

    Args:
        sdf_points (torch.Tensor): Query points in the voxel grid, shape [num_points, 3].
        sdf_values (torch.Tensor): Ground truth SDF values for the query points, shape [num_points].
        cone_params (torch.Tensor): Cone parameters, shape [num_cones, 8].
        
    Returns:
        torch.Tensor: The inside coverage loss.
    """
    # Filter for inside or on-surface points (SDF <= 0 in ground truth)
    inside_mask = sdf_values <= 0
    inside_points = sdf_points[inside_mask]
    inside_sdf_values = sdf_values[inside_mask]

    if inside_points.shape[0] == 0:  # No valid inside points
        return torch.tensor(0.0, device=sdf_points.device)

    # Compute the SDF of these points w.r.t. cones
    cone_sdf = determine_cone_sdf(inside_points, cone_params)  # Shape: [num_inside_points, num_cones]

    # Compute the minimum SDF across all cones for each point
    min_sdf, _ = torch.min(cone_sdf, dim=1)  # Shape: [num_inside_points]

    # Penalize uncovered points (cone SDF > ground truth SDF)
    uncovered_loss = torch.mean(torch.relu(min_sdf - inside_sdf_values))  # Positive mismatch penalization

    # Optional: Penalize excessive coverage (cone SDF < -1.0)
    over_coverage_loss = torch.mean(torch.relu(-min_sdf - 1.0))  # Penalize points far inside the cones

    # Combine losses with weights
    total_loss = uncovered_loss + 0.1 * over_coverage_loss

    return total_loss

def penalize_large_cones(cone_params):
    """
    Penalize spheres with large radii to encourage fitting finer features.

    Args:
        sphere_params (torch.Tensor): Sphere parameters (centers and radii).
        weight (float): Penalty weight for large spheres.

    Returns:
        torch.Tensor: Penalty for large radii.
    """
    cone_radii = cone_params[:, :, 3]
    cone_height = cone_params[:, :, 4]
    return torch.mean(cone_radii ** 2 + cone_height ** 2)  # Penalizes large radii

def cone_pairwise_difference_loss(cone_params):
    radii = cone_params[..., 3]
    heights = cone_params[..., 4]
    # Pairwise differences between radii and heights
    radii_diff = torch.cdist(radii.unsqueeze(-1), radii.unsqueeze(-1))
    heights_diff = torch.cdist(heights.unsqueeze(-1), heights.unsqueeze(-1))
    # Minimize the inverse of differences (penalize small differences)
    diversity_loss = 1.0 / (radii_diff + 1e-6).mean() + 1.0 / (heights_diff + 1e-6).mean()
    return diversity_loss

def height_diversity_loss(cone_params):
    heights = cone_params[..., 4]  # Extract cone heights
    mean_height = torch.mean(heights)
    diversity_loss = torch.mean((heights - mean_height) ** 2)  # Variance of heights
    return diversity_loss  # Penalize low variance (encourage variety)

# Sphere Loss Functions

def calculate_huber_loss(predictions, targets, delta=1.0):
    """
    Compute the Huber loss.

    Args:
        predictions (torch.Tensor): Predicted values.
        targets (torch.Tensor): Ground truth values.
        delta (float): Threshold parameter for the Huber loss.

    Returns:
        torch.Tensor: The computed Huber loss.
    """
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
    
    # Calculate overlap
    overlap_matrix = torch.relu(radii_matrix - dist_matrix)
    
    # Sum of squared overlaps (excluding self-overlap)
    overlap_loss = torch.sum(overlap_matrix ** 2) - torch.sum(torch.diag(overlap_matrix ** 2))
    
    return overlap_loss

def calculate_coverage_loss(voxel_data, sphere_params, sdf_points, sdf_values, sphere_sdf):
    # Get sphere parameters
    sphere_centers = sphere_params[:, :3]
    sphere_radii = sphere_params[:, 3]
    
    # Calculate SDF for each sphere at query points
    # sphere_sdf = determine_sphere_sdf(sdf_points, sphere_params)
    
    # Coverage loss: penalize distances from surface points to the spheres' zero level set
    # Use the minimum SDF value per query point across all spheres
    # min_sdf, _ = torch.min(sphere_sdf, dim=1)
    coverage_loss = torch.mean(torch.abs(sphere_sdf - sdf_values))
    
    # Uniformity loss: ensure spheres are evenly distributed
    min_distances, _ = torch.min(torch.norm(sdf_points[:, None, :] - sphere_centers[None, :, :], dim=-1), dim=1)
    uniformity_loss = torch.std(min_distances)
    
    # Regularization: control sphere size and avoid overlaps
    sphere_size_loss = torch.mean(torch.abs(sphere_radii))
    overlap_loss = calculate_overlap_loss(sphere_params)
    
    # Combine losses
    total_coverage_loss = coverage_loss + 0.1 * uniformity_loss + 0.01 * (sphere_size_loss + overlap_loss)
    return total_coverage_loss

def calculate_inside_coverage_loss(sdf_points, sdf_values, sphere_params):
    """
    Penalize lack of coverage for inside points of the voxel grid.
    
    Args:
        sdf_points (torch.Tensor): Query points in the voxel grid.
        sdf_values (torch.Tensor): Ground truth SDF values for the query points.
        sphere_params (torch.Tensor): Sphere parameters (centers and radii).
        
    Returns:
        torch.Tensor: The inside coverage loss.
    """
    # Get inside points (SDF values < 0)
    inside_mask = sdf_values < 0
    inside_points = sdf_points[inside_mask]
    
    if inside_points.shape[0] == 0:  # No inside points
        return torch.tensor(0.0, device=sdf_points.device)

    # Calculate SDF for these points w.r.t. spheres
    sphere_sdf = determine_sphere_sdf(inside_points, sphere_params)
    
    # Minimum SDF across all spheres for each inside point
    min_sdf, _ = torch.min(sphere_sdf, dim=1)
    
    # Penalize points with SDF > 0 (not covered by spheres)
    uncovered_loss = torch.mean(torch.relu(min_sdf))  # relu ensures only positive penalties
    
    return uncovered_loss

def calculate_graded_outside_loss(sphere_params, voxel_bounds, buffer=2.0, penalty_scale=2.0):
    """
    Penalize spheres that extend outside the voxel volume with graded penalties.

    Args:
        sphere_params (torch.Tensor): Sphere parameters (centers and radii).
        voxel_bounds (tuple): Min and max bounds of the voxel grid as ((xmin, ymin, zmin), (xmax, ymax, zmax)).
        buffer (float): Allowable extension beyond the bounds without penalty.
        penalty_scale (float): Scale factor for the penalty.

    Returns:
        torch.Tensor: The graded outside loss.
    """
    sphere_centers = sphere_params[:, :3]
    sphere_radii = sphere_params[:, 3]
    
    # Voxel bounds
    (xmin, ymin, zmin), (xmax, ymax, zmax) = voxel_bounds

    # Compute distances outside the bounds
    outside_xmin = torch.clamp(xmin - (sphere_centers[:, 0] - sphere_radii) - buffer, min=0)
    outside_ymin = torch.clamp(ymin - (sphere_centers[:, 1] - sphere_radii) - buffer, min=0)
    outside_zmin = torch.clamp(zmin - (sphere_centers[:, 2] - sphere_radii) - buffer, min=0)

    outside_xmax = torch.clamp((sphere_centers[:, 0] + sphere_radii) - xmax - buffer, min=0)
    outside_ymax = torch.clamp((sphere_centers[:, 1] + sphere_radii) - ymax - buffer, min=0)
    outside_zmax = torch.clamp((sphere_centers[:, 2] + sphere_radii) - zmax - buffer, min=0)

    # Apply a graded penalty (quadratic penalty for now)
    penalty_x = outside_xmin ** 2 + outside_xmax ** 2
    penalty_y = outside_ymin ** 2 + outside_ymax ** 2
    penalty_z = outside_zmin ** 2 + outside_zmax ** 2

    # Combine penalties and scale
    outside_loss = penalty_scale * torch.mean(penalty_x + penalty_y + penalty_z)

    return outside_loss

def calculate_size_diversity_loss(sphere_params):
    """
    Encourage diversity in sphere sizes by penalizing uniform radii.

    Args:
        sphere_params (torch.Tensor): Sphere parameters (centers and radii).

    Returns:
        torch.Tensor: Size diversity loss.
    """
    sphere_radii = sphere_params[:, 3]
    return -torch.std(sphere_radii)  # Negative standard deviation encourages variety

def penalize_large_spheres(sphere_params):
    """
    Penalize spheres with large radii to encourage fitting finer features.

    Args:
        sphere_params (torch.Tensor): Sphere parameters (centers and radii).
        weight (float): Penalty weight for large spheres.

    Returns:
        torch.Tensor: Penalty for large radii.
    """
    sphere_radii = sphere_params[:, 3]
    return torch.mean(sphere_radii ** 2)  # Penalizes large radii



#Incoming Change

# def cone_overlap_loss(cone_params):
#     batch_size, num_cones, _ = cone_params.shape
#     cone_centers = cone_params[:, :, :3]
#     cone_radii = cone_params[:, :, 3]
#     cone_heights = cone_params[:, :, 4]
    
#     overlap_loss = 0.0
    
#     for i in range(num_cones):
#         for j in range(i + 1, num_cones):
#             center_i = cone_centers[:, i, :]
#             center_j = cone_centers[:, j, :]
#             radius_i = cone_radii[:, i]
#             radius_j = cone_radii[:, j]
#             height_i = cone_heights[:, i]
#             height_j = cone_heights[:, j]
            
#             # Calculate the distance between the centers of the cones
#             distance = torch.norm(center_i - center_j, dim=-1)
            
#             # Calculate the overlap between the cones
#             overlap = torch.max(torch.tensor(0.0).to(cone_params.device), radius_i + radius_j - distance)
            
#             # Add the overlap to the total overlap loss
#             overlap_loss += overlap.mean()
    
#     return overlap_loss / (num_cones * (num_cones - 1) / 2)

# def calculate_inside_cone_coverage_loss(sdf_points, sdf_values, sphere_params):
#     """
#     Penalize lack of coverage for inside points of the voxel grid.
    
#     Args:
#         sdf_points (torch.Tensor): Query points in the voxel grid.
#         sdf_values (torch.Tensor): Ground truth SDF values for the query points.
#         sphere_params (torch.Tensor): Sphere parameters (centers and radii).
        
#     Returns:
#         torch.Tensor: The inside coverage loss.
#     """
#     # Get inside points (SDF values < 0)
#     inside_mask = sdf_values < 0
#     inside_points = sdf_points[inside_mask]
    
#     if inside_points.shape[0] == 0:  # No inside points
#         return torch.tensor(0.0, device=sdf_points.device)

#     # Calculate SDF for these points w.r.t. spheres
#     sphere_sdf = determine_cone_sdf(inside_points, sphere_params)
    
#     # Minimum SDF across all spheres for each inside point
#     min_sdf, _ = torch.min(sphere_sdf, dim=1)
    
#     # Penalize points with SDF > 0 (not covered by spheres)
#     uncovered_loss = torch.mean(torch.relu(min_sdf))  # relu ensures only positive penalties
    
#     return uncovered_loss

# def penalize_large_cones(cone_params):
#     """
#     Penalize cones with large radii and height to encourage fitting finer features.

#     Args:
#         cone_params (torch.Tensor): Cone parameters (centers and radii).
#         weight (float): Penalty weight for large cones.

#     Returns:
#         torch.Tensor: Penalty for large radii and height.
#     """
#     cone_radii = cone_params[:, :, 3]
#     cone_height = cone_params[:, :, 4]
#     return torch.mean(cone_radii ** 2 + cone_height ** 2)  # Penalizes large radii

# def cone_size_diversity_loss(cone_params):
#     radii = cone_params[..., 3]
#     heights = cone_params[..., 4]
#     # Penalize uniform radii and heights
#     return 5 * torch.std(radii) + 5 * torch.std(heights)

# def cone_pairwise_difference_loss(cone_params):
#     radii = cone_params[..., 3]
#     heights = cone_params[..., 4]
#     # Pairwise differences between radii and heights
#     radii_diff = torch.cdist(radii.unsqueeze(-1), radii.unsqueeze(-1))
#     heights_diff = torch.cdist(heights.unsqueeze(-1), heights.unsqueeze(-1))
#     # Minimize the inverse of differences (penalize small differences)
#     diversity_loss = 1.0 / (radii_diff + 1e-6).mean() + 1.0 / (heights_diff + 1e-6).mean()
#     return diversity_loss



#END OF CHANGE

def combined_fit_loss(combined_sdf, sdf_values):
    """
    Penalize the mismatch between the combined SDF and ground truth SDF.
    """

    print(f"        Combined SDF Shape: {combined_sdf.shape}")
    print(f"        SDF Values Shape: {sdf_values.shape}")

    reduced_sdf, _ = torch.min(combined_sdf, dim=2)  # Shape: [batch_size, num_points]
    reduced_sdf = reduced_sdf.squeeze(0)  # Remove batch dimension, shape: [num_points]

    # Compute mean squared error
    return torch.mean((reduced_sdf - sdf_values) ** 2)


### INCOMING CHANGE

# cylinder loss functions
def penalize_large_cylinders(cylinder_params):
    """
    Penalize cylinders with large radii and height to encourage fitting finer features.

    Args:
        cylinder_params (torch.Tensor): Cylinder parameters (centers, axes, radii, and height).

    Returns:
        torch.Tensor: Penalty for large radii and height.
    """
    cylinder_radii = cylinder_params[:,6]
    cylinder_height = cylinder_params[:,7]
    return torch.mean(cylinder_radii ** 2 + cylinder_height ** 2)
###END OF INCOMING CHANGE