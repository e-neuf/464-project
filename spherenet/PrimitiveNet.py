import torch
import torch.nn as nn
import numpy as np
import os
import trimesh
from scipy.ndimage import gaussian_filter
from skimage.transform import resize
from Loss import calculate_coverage_loss, calculate_huber_loss, calculate_inside_coverage_loss, calculate_graded_outside_loss, calculate_size_diversity_loss, penalize_large_spheres, calculate_inside_cone_coverage_loss, cone_overlap_loss, penalize_large_cones, calculate_overlap_loss, cone_size_diversity_loss, cone_pairwise_difference_loss, cone_overlap_loss, combined_fit_loss,height_diversity_loss, calculate_graded_outside_loss_cone_sdf
from ConeNet import ConeNet, visualize_cones
from SphereNet import SphereNet, visualise_spheres
from sklearn.cluster import KMeans
from SDF import determine_cone_sdf, determine_sphere_sdf
from MetaNet import MetaNet
import torch.nn.functional as F
import random
import time

def bsmin(a, dim, k=22.0, keepdim=False):
    dmix = -torch.logsumexp(-k * a, dim=dim, keepdim=keepdim) / k
    return dmix

def visualize_primitives(points, values, sphere_params, cone_params, save_path=None):
    """
    Visualize spheres and cones using Trimesh.
    
    Args:
        sphere_params (np.ndarray): Parameters for spheres, shape [num_spheres, 4].
                                     Each sphere is represented as [x, y, z, r].
        cone_params (np.ndarray): Parameters for cones, shape [num_cones, 8].
                                   Each cone is represented as [x, y, z, r, h, dx, dy, dz].
        save_path (str, optional): Path to save the visualization as an OBJ file.
                                   If None, the visualization is shown interactively.
    """
    # Convert tensors to NumPy arrays if needed
    if isinstance(sphere_params, torch.Tensor):
        sphere_params = sphere_params.cpu().numpy()
    if isinstance(cone_params, torch.Tensor):
        cone_params = cone_params.cpu().numpy()

    # Handle extra dimensions in cone_params
    if len(cone_params.shape) == 3:  # If batch dimension exists
        cone_params = cone_params.squeeze(0)  # Remove batch dimension

    # Sanity check for invalid values
    assert np.all(np.isfinite(sphere_params)), "Sphere parameters contain invalid values!"
    assert np.all(np.isfinite(cone_params)), "Cone parameters contain invalid values!"

    scene = trimesh.Scene()

    # Add spheres to the scene
    for center, radius in zip(sphere_params[:, :3], sphere_params[:, 3]):
        try:
            radius = float(radius)
            sphere = trimesh.creation.icosphere(radius=abs(radius), subdivisions=3)
            sphere.apply_translation(center)

            # colour = [255,0,0,255]
            # sphere.visual.face_colors = colour

            scene.add_geometry(sphere)
        except Exception as e:
            print(f"Error creating sphere with center={center}, radius={radius}: {e}")

    # Add cones to the scene
    for i, (center, radius, height, orientation) in enumerate(zip(
        cone_params[:, :3],
        cone_params[:, 3],
        cone_params[:, 4],
        cone_params[:, 5:8]
    )):
        try:
            # Ensure radius and height are scalars
            radius = float(radius)
            height = float(height)

            # Default orientation if invalid
            if np.linalg.norm(orientation) == 0:
                orientation = np.array([0, 0, 1])
            orientation = orientation / np.linalg.norm(orientation)

            # Create cone
            cone = trimesh.creation.cone(radius=abs(radius), height=abs(height))

            # Align the cone orientation
            z_axis = np.array([0, 0, 1])  # Default cone direction
            rotation_matrix = trimesh.geometry.align_vectors(z_axis, orientation)
            cone.apply_transform(rotation_matrix)

            # Translate cone to its center
            cone.apply_translation(center)

            # color = [0, 0, 255, 255]
            # cone.visual.face_colors = color

            scene.add_geometry(cone)
        except Exception as e:
            print(f"Error creating cone {i} with center={center}, radius={radius}, height={height}, orientation={orientation}: {e}")

    inside_points = points[values < 0]
    inside_points = trimesh.points.PointCloud(inside_points)
    inside_points.colors = [0, 0, 255, 255]  # Blue color for inside points
    scene.add_geometry([inside_points])

    # Show the scene interactively
    if save_path is None:
        scene.show()
    else:
        # Save to a file
        scene.show()
        scene.export(save_path)
        print(f"Visualization")

def visualise_sdf(points, values):
    inside_points = points[values < 0]
    outside_points = points[values > 0]
    inside_points = trimesh.points.PointCloud(inside_points)
    outside_points = trimesh.points.PointCloud(outside_points)
    inside_points.colors = [0, 0, 255, 255]  # Blue color for inside points
    outside_points.colors = [255, 0, 0, 255]  # Red color for outside points
    scene = trimesh.Scene()
    scene.add_geometry([inside_points, outside_points])
    scene.show()

def voxel_to_mesh(voxel_data):
    # Convert voxel data to a mesh representation
    voxel_data = voxel_data.cpu().numpy()
    mesh = trimesh.voxel.ops.matrix_to_marching_cubes(voxel_data, pitch=1.0)
    return mesh

def preprocess_voxel_data(voxel_data, target_shape=(64, 64, 64), sigma=1.0):
    # Normalize the voxel data
    voxel_data = voxel_data / voxel_data.max()

    # Apply Gaussian smoothing
    voxel_data = gaussian_filter(voxel_data, sigma=sigma)

    # Resample the voxel data
    voxel_data = resize(voxel_data, target_shape, mode='constant', anti_aliasing=True)

    return voxel_data

def visualise_voxels(voxel_data):
    """
    Visualize voxel data using Trimesh.

    Args:
        voxel_data (torch.Tensor or np.ndarray): A 3D boolean tensor/array representing voxel occupancy.
    """
    # Convert PyTorch tensor to NumPy array if needed
    if isinstance(voxel_data, torch.Tensor):
        voxel_data = voxel_data.cpu().numpy()

    # Ensure voxel_data is boolean
    voxel_data = voxel_data.astype(bool)

    # Create a VoxelGrid from the boolean array
    voxel_grid = trimesh.voxel.VoxelGrid(voxel_data)

    # Convert voxel grid to a mesh
    mesh = voxel_grid.as_boxes()

    # Create a scene and add the mesh
    scene = trimesh.Scene()
    scene.add_geometry(mesh)

    # Show the scene
    scene.show()

def visualize_cones_small(points, values, cone_params, save_path=None):
    """
    Visualize cones using their parameters and SDF data.

    Args:
        points (torch.Tensor): Points in the 3D space, shape [num_points, 3].
        values (torch.Tensor): SDF values for the points, shape [num_points].
        cone_params (torch.Tensor): Parameters for cones, shape [x, 8].
        save_path (str, optional): Path to save the visualization. Defaults to None.
    """
    # Ensure cone_params is of shape [x, 8]
    cone_params = cone_params.squeeze(0).cpu().detach().numpy()  # Handle any extra batch dimensions safely
    cone_centers = cone_params[:, :3]
    cone_radii = np.abs(cone_params[:, 3])
    cone_heights = np.abs(cone_params[:, 4])
    cone_orientations = cone_params[:, 5:8]
    scene = trimesh.Scene()

    for i in range(cone_centers.shape[0]):
        center = cone_centers[i]
        radius = cone_radii[i]
        height = cone_heights[i]
        orientation = cone_orientations[i]

        # Ensure radius and height are scalar values
        radius = float(radius)
        height = float(height)

        # Extract points within the region of the cone
        mask = np.linalg.norm(points.cpu().detach().numpy() - center, axis=1) < radius
        region_points = points[mask].cpu().detach().numpy()

        # Normalize the orientation vector
        if np.linalg.norm(orientation) < 1e-6:
            orientation = np.array([0, 0, 1])  # Default orientation if invalid
        orientation = orientation / np.linalg.norm(orientation)

        # Create the cone
        cone = trimesh.creation.cone(radius=radius, height=height)

        # Compute the rotation matrix to align the cone with the orientation vector
        z_axis = np.array([0, 0, 1])
        rotation_axis = np.cross(z_axis, orientation)
        if np.linalg.norm(rotation_axis) < 1e-6:
            rotation_matrix = np.eye(4)  # No rotation needed
        else:
            rotation_angle = np.arccos(np.dot(z_axis, orientation))
            rotation_matrix = trimesh.transformations.rotation_matrix(rotation_angle, rotation_axis)

        # Apply the rotation and translation to the cone
        cone.apply_transform(rotation_matrix)
        cone.apply_translation(center)
        scene.add_geometry(cone)

    # Visualize inside and outside points
    inside_points = points[values < 0].cpu().detach().numpy()
    outside_points = points[values > 0].cpu().detach().numpy()
    if len(inside_points) > 0:
        inside_points = trimesh.points.PointCloud(inside_points)
        inside_points.colors = np.array([[0, 0, 255, 255]] * len(inside_points.vertices))  # Blue color for inside points
        scene.add_geometry([inside_points])
    # if len(outside_points) > 0:
    #     outside_points = trimesh.points.PointCloud(outside_points)
    #     outside_points.colors = np.array([[255, 0, 0, 255]] * len(outside_points.vertices))  # Red color for outside points
    #     scene.add_geometry([outside_points])

    # Save or show the scene
    if save_path is not None:
        scene.export(save_path)
    scene.show()

def run_spherenet(sphere_optimizer, sphere_model, voxel_data, points, values, device, i):
        sphere_optimizer.zero_grad()
        sphere_sdf, sphere_params = sphere_model(
            voxel_data.unsqueeze(0), points
        )

        sphere_sdf = bsmin(sphere_sdf, dim=-1).to(device)

        sphere_mseloss = torch.mean((sphere_sdf - values) ** 2)
        

        # coverage_loss = calculate_coverage_loss(points, sphere_params, points, values, sphere_sdf)

        # overlap_loss = calculate_overlap_loss(sphere_params)

        # huber_loss = calculate_huber_loss(sphere_sdf, values, delta=0.3)

        inside_coverage_loss = calculate_inside_coverage_loss(points, values, sphere_params)

        outside_loss = calculate_graded_outside_loss(sphere_params, ((0,0,0), (64,64,64)), buffer=0.3, penalty_scale=2.0)

        large_sphere_penalty = penalize_large_spheres(sphere_params)

        print(f"    Sphere MSL loss: {sphere_mseloss.item()}")
        # print(f"    Combined coverage loss: {combined_coverage_loss_number.item()}")
        # print(f"    Combined overlap loss: {combined_overlap_loss_number.item()}")
        print(f"    Inside coverage loss: {inside_coverage_loss.item()}")
        print(f"    Outside loss: {outside_loss.item()}")
        # # print(f"    Diversity loss: {diversity_loss.item()}")
        print(f"    Large sphere penalty: {large_sphere_penalty.item()}")
        # print(f"Huber loss: {huber_loss.item()}")
        # print(f"Coverage loss: {coverage_loss.item()}")
        # print(f"Overlap loss: {overlap_loss.item()}")

        loss = sphere_mseloss + 0.5*inside_coverage_loss + 0.5*outside_loss + 0.4*large_sphere_penalty
        # loss = sphere_mseloss
        
        loss.backward(retain_graph=True)
        sphere_optimizer.step()
        print(f"Sphere Iteration {i}, Loss: {loss.item()}")

        return sphere_params

def run_conenet(optimizer, model, voxel_data, points, values, device, i, num_epochs):
    optimizer.zero_grad()
    cone_sdf, cone_params = model(
        voxel_data.unsqueeze(0), points
    )
    
    cone_sdf = bsmin(cone_sdf, dim=-1).to(device)

    mseloss = torch.mean((cone_sdf - values) ** 2)

    inside_coverage_loss = calculate_inside_cone_coverage_loss(points, values, cone_params)

    outside_loss = calculate_graded_outside_loss_cone_sdf(cone_params, points, values, penalty_scale=2.0)

    # huber_loss = calculate_huber_loss(cone_sdf, values, delta=0.3)


    # large_cone_penalty = penalize_large_cones(cone_params)

    # overlap_loss = cone_overlap_loss(cone_params)

    # diversity_loss = cone_size_diversity_loss(cone_params)

    # rotation_loss = compute_cone_rotation_loss(points, cone_params)

    loss = mseloss
    loss += (0.5 * inside_coverage_loss)
    loss += (0.5 * outside_loss)
    # loss += (0.5 * huber_loss)
    # loss += 0.2 * large_cone_penalty
    # loss += (0.1 * overlap_loss)
    # loss += 0.1 * diversity_loss
    # loss += 0.1 * rotation_loss      

    print ("    mse loss: ", mseloss.item())
    print ("    inside_coverage_loss: ", inside_coverage_loss.item())
    # print ("    overlap_loss: ", overlap_loss.item())
    print ("    outside_loss: ", outside_loss.item())
    # print ("    large_cones_loss: ", large_cones_loss.item())
    # print ("    height_loss : ", height_loss.item())
    # print ("    rotation_loss : ", rotation_loss.item())
    # print ("    huber_loss: ", huber_loss.item())

    loss.backward()
    optimizer.step()
    print(f"Cone Iteration {i}, Loss: {loss.item()}\n")
    return cone_params

def run_training_loop(output_dir, model_name, iterations, num_primitives):
    startTime = time.time()

    torch.autograd.set_detect_anomaly(True)

    num_cones = num_primitives
    num_spheres = num_primitives
    num_epochs = iterations

    dataset_path = "./reference_models_processed"
    name = model_name

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    path = os.path.join(dataset_path, name, "voxel_and_sdf.npz")

    # Load the voxel data from the .npz file
    data = np.load(os.path.join(dataset_path, name, "voxel_and_sdf.npz"))
    print(data.files)  # Inspect the contents of the .npz file

    voxel_data = data["voxels"]
    centroid = data["centroid"]
    scale = data["scale"]

    # Preprocess the voxel data
    voxel_data = preprocess_voxel_data(voxel_data)
    voxel_data = torch.from_numpy(voxel_data).float().to(device)

    # Convert voxel data to mesh
    reference_model = voxel_to_mesh(voxel_data)

    # Load other necessary data
    points = data["sdf_points"]
    values = data["sdf_values"]

    # visualise_sdf(points, values)

    # visualise_voxels(voxel_data)

    points = torch.from_numpy(points).float().to(device)
    values = torch.from_numpy(values).float().to(device)

    sphere_model = SphereNet(num_spheres=num_spheres).to(device)
    sphere_optimizer = torch.optim.Adam(sphere_model.parameters(), lr=0.0005)

    cone_model = ConeNet(num_cones=num_cones).to(device)
    cone_optimizer = torch.optim.Adam(cone_model.parameters(), lr=0.0005)

    for i in range(num_epochs):
        sphere_params = run_spherenet(sphere_optimizer, sphere_model, voxel_data, points, values, device, i)
        cone_params = run_conenet(cone_optimizer, cone_model, voxel_data, points, values, device, i, num_epochs)
    
    endTime = time.time()

    print(f"            Total time taken: {endTime - startTime} seconds")

    sphere_params = sphere_params.cpu().detach()
    cone_params = cone_params.cpu().detach()

    # sphere_sdf = determine_sphere_sdf(points, sphere_params)
    # cone_sdf = determine_cone_sdf(points, cone_params)

    
    # sphere_errors = torch.mean((sphere_sdf - values.unsqueeze(1)) ** 2, dim=0)  # [num_spheres]
    # cone_errors = torch.mean((cone_sdf.squeeze(0) - values.unsqueeze(1)) ** 2, dim=0)  # [num_cones]

    # sphere_paramms_with_errors = torch.cat((sphere_params, sphere_errors.unsqueeze(1)), dim=1)
    # cone_paramms_with_errors = torch.cat((cone_params.squeeze(0), cone_errors.unsqueeze(1)), dim=1)


    # #print max and min values for errors
    # print("     Sphere Errors Max: ", sphere_errors.max().item())
    # print("     Sphere Errors Min: ", sphere_errors.min().item())
    # print("     Cone Errors Max: ", cone_errors.max().item())
    # print("     Cone Errors Min: ", cone_errors.min().item())

    # sphere_pruning_threshold = 0.5
    # cone_pruning_threshold = 0.05

    # sphere_mask = sphere_paramms_with_errors[:, -1] < sphere_pruning_threshold
    # cone_mask = cone_paramms_with_errors[:, -1] < cone_pruning_threshold

    # pruned_sphere_params = sphere_paramms_with_errors[sphere_mask]
    # pruned_cone_params = cone_paramms_with_errors[cone_mask]

    # pruned_sphere_params = pruned_sphere_params[:, :-1]
    # pruned_cone_params = pruned_cone_params[:, :-1]

    # visualise_spheres(points, values, sphere_params, reference_model=None, save_path=None)
    # visualise_spheres(points, values, pruned_sphere_params, reference_model=None, save_path=None)

    os.makedirs(output_dir, exist_ok=True)

    # visualize_cones_small(points, values, cone_params, save_path=os.path.join(output_dir, f"{name}_cones.obj"))
    # visualize_cones_small(points, values, pruned_cone_params, save_path=os.path.join(output_dir, f"{name}_spheres.obj"))
    # visualize_primitives(pruned_sphere_params, pruned_cone_params, save_path=os.path.join(output_dir, f"{name}_combined.obj"")  
    
    np.save(os.path.join(output_dir, f"{name}_sphere_params.npy"), sphere_params.cpu().detach().numpy())
    np.save(os.path.join(output_dir, f"{name}_cone_params.npy"), cone_params.cpu().detach().numpy())

    # visualise_spheres(points, values, sphere_params, reference_model=None, save_path=os.path.join(output_dir, f"{name}_spheres.obj"))
    # visualize_cones(points, values, cone_params, save_path=os.path.join(output_dir, f"{name}_cones.obj"))

    torch.save(sphere_model.state_dict(), os.path.join(output_dir, f"{name}_sphere_model.pth"))
    torch.save(cone_model.state_dict(), os.path.join(output_dir, f"{name}_cone_model.pth"))

def main():
    output_dir = "./output"
    models = [
        'dog', 
        'hand', 
        'pot', 
        'rod', 
        'sofa'
    ]
    
    iterations = 200
    num_primitives = 256

    for model in models:
        print (f"Running training loop for {model}")
        run_training_loop(output_dir, model, iterations, num_primitives)
        print (f"Finished training loop for {model}")
        print("############################################")

if __name__ == "__main__":
    main()