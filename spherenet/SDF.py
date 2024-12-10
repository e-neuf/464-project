import torch
import torch.nn.functional as F
import numpy as np

def determine_sphere_sdf(query_points, sphere_params):
    sphere_centers = sphere_params[:,:3]
    sphere_radii = sphere_params[:,3]
    vectors_points_to_centers = query_points[:, None, :] - sphere_centers[None, :, :]
    distance_points_to_centers = torch.norm(vectors_points_to_centers, dim=-1)
    sphere_sdf = distance_points_to_centers - sphere_radii
    return sphere_sdf

def determine_cone_sdf(query_points, cone_params):
    cone_centers = cone_params[:, :3]  
    cone_radii = cone_params[:, 3:4]  
    cone_heights = cone_params[:, 4:5]  
    cone_orientations = cone_params[:, 5:8]  

    # Normalize cone orientations
    cone_orientations = F.normalize(cone_orientations, dim=-1)  
    vectors_to_centers = query_points[:, None, :] - cone_centers[None, :, :]  

    # Project vectors onto the cone's orientation axis
    projections_onto_axis = torch.sum(vectors_to_centers * cone_orientations[None, :, :], dim=-1, keepdim=True)  

    # perpendicular distances to the cone's axis
    perpendicular_distances = torch.norm(vectors_to_centers - projections_onto_axis * cone_orientations[None, :, :], dim=-1, keepdim=True)  
    safe_heights = torch.clamp(cone_heights, min=1e-6)

    # Calculate normalized radius for points along the height
    normalized_height = torch.clamp(projections_onto_axis / safe_heights, min=0.0, max=1.0)  
    scaled_radius = normalized_height * cone_radii  # Radius at the query point's height

    # Compute SDF to cone surface
    slanted_surface_sdf = perpendicular_distances - scaled_radius  # Distance to slanted surface
    cap_sdf = projections_onto_axis - cone_heights  # Distance to cone cap
    base_sdf = -projections_onto_axis  # Distance to cone base

    combined_sdf = torch.max(torch.cat([slanted_surface_sdf, base_sdf, cap_sdf], dim=-1), dim=-1).values 

    return combined_sdf

def determine_cylinder_sdf(query_points, cylinder_params):
    centers = cylinder_params[:,:3]
    axes = cylinder_params[:,3:6]
    radii = cylinder_params[:,6]
    heights = cylinder_params[:,7]

    num_query_points = query_points.shape[0]
    num_cylinders = cylinder_params.shape[0]
    dist_to_cyl = torch.empty((num_query_points, num_cylinders))

    axes_normalized = F.normalize(axes)
    
    for i in range(num_cylinders):
        # project query points onto cylinder axis
        scalar_points = torch.linalg.vecdot(query_points - centers[i,:], axes_normalized[i,:] - centers[i,:])
        projection = scalar_points[:, np.newaxis] * axes_normalized[i,:]
        closest_points = centers[i,:] + projection
        
        # calcualte distance from query points to cylinder axis
        dist_to_axis = torch.linalg.vector_norm(query_points - closest_points, dim=1)
        point_is_within_radius = torch.le(dist_to_axis, radii[i])

        # calcuate the points on the top and the bottom of the cylinder axis
        interm = (heights[i]/2) / torch.linalg.vector_norm(axes[i,:])
        top = centers[i,:] + interm * axes[i,:] 
        bottom = centers[i,:] - interm * axes[i,:]

        # calculate distance from query points to cylinder top/bottom
        dist_to_top = torch.linalg.vector_norm(projection - top, dim=1)
        dist_to_bottom = torch.linalg.vector_norm(projection - bottom, dim=1)
        dist_to_height = torch.minimum(dist_to_top, dist_to_bottom)
        point_is_within_height = torch.le((dist_to_top + dist_to_bottom), heights[i])

        # use the appropriate distance for each query point
        point_is_inside = point_is_within_height & point_is_within_radius
        point_is_within_none = ~(point_is_within_height | point_is_within_radius)

        dist_to_cyl[point_is_within_height,i] = dist_to_axis[point_is_within_height]
        dist_to_cyl[point_is_within_radius,i] = dist_to_height[point_is_within_radius]
        inside = -1 * torch.minimum(torch.abs(dist_to_axis - radii[i]), dist_to_height)
        dist_to_cyl[point_is_inside,i] = inside[point_is_inside]
        dist_to_cyl[point_is_within_none,i] = torch.sqrt(dist_to_axis[point_is_within_none]**2 + dist_to_height[point_is_within_none]**2)
    
    return dist_to_cyl
