import torch
from SDF import determine_sphere_sdf

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

