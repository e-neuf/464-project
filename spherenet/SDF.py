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
    """
    Calculate the Signed Distance Function (SDF) values for a set of cones at given query points.

    Args:
        query_points (torch.Tensor): Query points in 3D space, shape [num_points, 3].
        cone_params (torch.Tensor): Parameters for cones, shape [num_cones, 8].
                                     Each cone has parameters [center_x, center_y, center_z, radius, height, dir_x, dir_y, dir_z].

    Returns:
        torch.Tensor: SDF values for each query point w.r.t each cone, shape [num_points, num_cones].
    """
    # Extract cone parameters
    cone_centers = cone_params[:, :3]  # Shape: [num_cones, 3]
    cone_radii = cone_params[:, 3:4]  # Shape: [num_cones, 1]
    cone_heights = cone_params[:, 4:5]  # Shape: [num_cones, 1]
    cone_orientations = cone_params[:, 5:8]  # Shape: [num_cones, 3]

    # Normalize cone orientations
    cone_orientations = F.normalize(cone_orientations, dim=-1)  # Shape: [num_cones, 3]

    # Compute vectors from query points to cone centers
    vectors_to_centers = query_points[:, None, :] - cone_centers[None, :, :]  # Shape: [num_points, num_cones, 3]

    # Project vectors onto the cone's orientation axis
    projections_onto_axis = torch.sum(vectors_to_centers * cone_orientations[None, :, :], dim=-1, keepdim=True)  # Shape: [num_points, num_cones, 1]

    # Compute perpendicular distances to the cone's axis
    perpendicular_distances = torch.norm(vectors_to_centers - projections_onto_axis * cone_orientations[None, :, :], dim=-1, keepdim=True)  # Shape: [num_points, num_cones, 1]

    # Handle cones with zero or near-zero height to prevent division by zero
    safe_heights = torch.clamp(cone_heights, min=1e-6)

    # Calculate normalized radius for points along the height
    normalized_height = torch.clamp(projections_onto_axis / safe_heights, min=0.0, max=1.0)  # Shape: [num_points, num_cones, 1]
    scaled_radius = normalized_height * cone_radii  # Radius at the query point's height

    # Compute SDF to cone surface
    slanted_surface_sdf = perpendicular_distances - scaled_radius  # Distance to slanted surface
    cap_sdf = projections_onto_axis - cone_heights  # Distance to cone cap
    base_sdf = -projections_onto_axis  # Distance to cone base

    # Combine SDF components
    combined_sdf = torch.max(torch.cat([slanted_surface_sdf, base_sdf, cap_sdf], dim=-1), dim=-1).values  # Shape: [num_points, num_cones]

    return combined_sdf