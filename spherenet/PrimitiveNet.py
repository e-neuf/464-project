import torch
import torch.nn as nn
import numpy as np
import os
import trimesh
from scipy.ndimage import gaussian_filter
from skimage.transform import resize
from Loss import  calculate_huber_loss, calculate_inside_coverage_loss, calculate_graded_outside_loss, penalize_large_spheres, calculate_inside_cone_coverage_loss, cone_overlap_loss, calculate_overlap_loss, cone_overlap_loss, calculate_graded_outside_loss_cone_sdf
from ConeNet import ConeNet, visualize_cones
from spherenet import SphereNet, visualise_spheres
from sklearn.cluster import KMeans
from SDF import determine_cone_sdf, determine_sphere_sdf
from MetaNet import MetaNet
import torch.nn.functional as F
import random
import time

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
    # Convert tensors to NumPy arrays 
    if isinstance(sphere_params, torch.Tensor):
        sphere_params = sphere_params.cpu().numpy()
    if isinstance(cone_params, torch.Tensor):
        cone_params = cone_params.cpu().numpy()

    # Handle extra dimensions in cone_params
    if len(cone_params.shape) == 3:  
        cone_params = cone_params.squeeze(0)  

    assert np.all(np.isfinite(sphere_params)), "Sphere parameters contain invalid values!"
    assert np.all(np.isfinite(cone_params)), "Cone parameters contain invalid values!"

    scene = trimesh.Scene()

    # Add spheres to the scene
    for center, radius in zip(sphere_params[:, :3], sphere_params[:, 3]):
        try:
            radius = float(radius)
            sphere = trimesh.creation.icosphere(radius=abs(radius), subdivisions=3)
            sphere.apply_translation(center)

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
            radius = float(radius)
            height = float(height)

            # Default orientation if invalid
            if np.linalg.norm(orientation) == 0:
                orientation = np.array([0, 0, 1])
            orientation = orientation / np.linalg.norm(orientation)

            cone = trimesh.creation.cone(radius=abs(radius), height=abs(height))

            # Align the cone orientation
            z_axis = np.array([0, 0, 1])  # Default cone direction
            rotation_matrix = trimesh.geometry.align_vectors(z_axis, orientation)
            cone.apply_transform(rotation_matrix)

            cone.apply_translation(center)

            scene.add_geometry(cone)
        except Exception as e:
            print(f"Error creating cone {i} with center={center}, radius={radius}, height={height}, orientation={orientation}: {e}")

    inside_points = points[values < 0]
    inside_points = trimesh.points.PointCloud(inside_points)
    inside_points.colors = [0, 0, 255, 255]  
    scene.add_geometry([inside_points])

    if save_path is None:
        scene.show()
    else:
        scene.show()
        scene.export(save_path)
        print(f"Visualization")


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
    cone_params = cone_params.squeeze(0).cpu().detach().numpy() 
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

        radius = float(radius)
        height = float(height)

        # Extract points within the region of the cone
        mask = np.linalg.norm(points.cpu().detach().numpy() - center, axis=1) < radius
        region_points = points[mask].cpu().detach().numpy()

        # Normalize the orientation vector
        if np.linalg.norm(orientation) < 1e-6:
            orientation = np.array([0, 0, 1])  
        orientation = orientation / np.linalg.norm(orientation)

        cone = trimesh.creation.cone(radius=radius, height=height)

        # Compute the rotation matrix to align the cone with the orientation vector
        z_axis = np.array([0, 0, 1])
        rotation_axis = np.cross(z_axis, orientation)
        if np.linalg.norm(rotation_axis) < 1e-6:
            rotation_matrix = np.eye(4)  #identity matrix
        else:
            rotation_angle = np.arccos(np.dot(z_axis, orientation))
            rotation_matrix = trimesh.transformations.rotation_matrix(rotation_angle, rotation_axis)

        cone.apply_transform(rotation_matrix)
        cone.apply_translation(center)
        scene.add_geometry(cone)

    inside_points = points[values < 0].cpu().detach().numpy()
    if len(inside_points) > 0:
        inside_points = trimesh.points.PointCloud(inside_points)
        inside_points.colors = np.array([[0, 0, 255, 255]] * len(inside_points.vertices))  
        scene.add_geometry([inside_points])

    if save_path is not None:
        scene.export(save_path)
    scene.show()


def bsmin(a, dim, k=22.0, keepdim=False):
    dmix = -torch.logsumexp(-k * a, dim=dim, keepdim=keepdim) / k
    return dmix

def visualise_sdf(points, values):
    inside_points = points[values < 0]
    outside_points = points[values > 0]
    inside_points = trimesh.points.PointCloud(inside_points)
    outside_points = trimesh.points.PointCloud(outside_points)
    inside_points.colors = [0, 0, 255, 255]  
    outside_points.colors = [255, 0, 0, 255]  
    scene = trimesh.Scene()
    scene.add_geometry([inside_points, outside_points])
    scene.show()

def preprocess_voxel_data(voxel_data, target_shape=(64, 64, 64), sigma=1.0):
    # Normalize the voxel data
    voxel_data = voxel_data / voxel_data.max()

    # Apply Gaussian smoothing
    voxel_data = gaussian_filter(voxel_data, sigma=sigma)

    # Resample the voxel data
    voxel_data = resize(voxel_data, target_shape, mode='constant', anti_aliasing=True)

    return voxel_data

def visualise_voxels(voxel_data):
    # Convert PyTorch tensor to NumPy array 
    if isinstance(voxel_data, torch.Tensor):
        voxel_data = voxel_data.cpu().numpy()

    voxel_data = voxel_data.astype(bool)
    voxel_grid = trimesh.voxel.VoxelGrid(voxel_data)
    mesh = voxel_grid.as_boxes()

    scene = trimesh.Scene()
    scene.add_geometry(mesh)
    scene.show()

def run_spherenet(sphere_optimizer, sphere_model, voxel_data, points, values, device, i):
        sphere_optimizer.zero_grad()
        sphere_sdf, sphere_params = sphere_model(
            voxel_data.unsqueeze(0), points
        )

        sphere_sdf = bsmin(sphere_sdf, dim=-1).to(device)
        sphere_mseloss = torch.mean((sphere_sdf - values) ** 2)
        
        inside_coverage_loss = calculate_inside_coverage_loss(points, values, sphere_params)
        outside_loss = calculate_graded_outside_loss(sphere_params, ((0,0,0), (64,64,64)), buffer=0.3, penalty_scale=2.0)
        large_sphere_penalty = penalize_large_spheres(sphere_params)

        print(f"    Sphere MSL loss: {sphere_mseloss.item()}")
        print(f"    Inside coverage loss: {inside_coverage_loss.item()}")
        print(f"    Outside loss: {outside_loss.item()}")
        print(f"    Large sphere penalty: {large_sphere_penalty.item()}")

        loss = sphere_mseloss + 0.5*inside_coverage_loss + 0.5*outside_loss + 0.4*large_sphere_penalty

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

    loss = mseloss
    loss += (0.5 * inside_coverage_loss)
    loss += (0.5 * outside_loss)

    print ("    mse loss: ", mseloss.item())
    print ("    inside_coverage_loss: ", inside_coverage_loss.item())
    print ("    outside_loss: ", outside_loss.item())

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
    print(data.files)  

    voxel_data = data["voxels"]

    voxel_data = preprocess_voxel_data(voxel_data)
    voxel_data = torch.from_numpy(voxel_data).float().to(device)

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

    os.makedirs(output_dir, exist_ok=True)

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
        # 'hand', 
        # 'pot', 
        # 'rod', 
        # 'sofa'
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