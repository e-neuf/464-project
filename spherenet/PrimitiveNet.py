import torch
import torch.nn as nn
import numpy as np
import os
import trimesh
from scipy.ndimage import gaussian_filter
from skimage.transform import resize
from Loss import calculate_coverage_loss, calculate_huber_loss, calculate_inside_coverage_loss, calculate_graded_outside_loss, calculate_size_diversity_loss, penalize_large_spheres, calculate_inside_cone_coverage_loss, cone_overlap_loss, penalize_large_cones, calculate_overlap_loss, cone_size_diversity_loss, cone_pairwise_difference_loss, cone_overlap_loss, combined_fit_loss
from ConeNet import ConeNet, visualize_cones
from SphereNet import SphereNet, visualise_spheres
from sklearn.cluster import KMeans
from SDF import determine_cone_sdf, determine_sphere_sdf
from MetaNet import MetaNet
import torch.nn.functional as F
import random


def bsmin(a, dim, k=22.0, keepdim=False):
    dmix = -torch.logsumexp(-k * a, dim=dim, keepdim=keepdim) / k
    return dmix

def visualize_primitives(sphere_params, cone_params, save_path=None):
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
            scene.add_geometry(cone)
        except Exception as e:
            print(f"Error creating cone {i} with center={center}, radius={radius}, height={height}, orientation={orientation}: {e}")

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

def initialize_cone_centers(voxel_points, values, num_cones, device):
    kmeans = KMeans(n_clusters=num_cones)

    voxel_points = voxel_points[values < 0]
    kmeans.fit(voxel_points)
    centers = kmeans.cluster_centers_
    return torch.tensor(centers, dtype=torch.float32).to(device)

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
        
        loss.backward(retain_graph=True)
        sphere_optimizer.step()
        print(f"Iteration {i}, Loss: {loss.item()}")

        return sphere_params

def run_conenet(optimizer, model, voxel_data, points, values, device, i, initial_centers):
    optimizer.zero_grad()
    cone_sdf, cone_params = model(
        voxel_data.unsqueeze(0), points, initial_centers
    )
    
    cone_sdf = bsmin(cone_sdf, dim=-1).to(device)

    mseloss = torch.mean((cone_sdf - values) ** 2)

    inside_coverage_loss = calculate_inside_cone_coverage_loss(points, values, cone_params)

    # overlap_loss = cone_overlap_loss(cone_params)

    outside_loss = calculate_graded_outside_loss(cone_params, ((0,0,0), (64,64,64)), buffer=0.3, penalty_scale=2.0)

    large_cones_loss = penalize_large_cones(cone_params)

    diversity_loss = cone_size_diversity_loss(cone_params)

    # pairwise_loss = cone_pairwise_difference_loss(cone_params)
    
    # Bonus: Design additional losses that helps to achieve a better result.
    # mseloss = 0
    # 0.5 * overlap_loss + 0.5 * inside_coverage_loss + 0.5 * outside_loss + 0.4 * large_cones_loss
    # loss = mseloss + 0.5 * diversity_loss + 0.4 * large_cones_loss
    loss = mseloss + diversity_loss + 0.5*inside_coverage_loss + outside_loss + 0.5*large_cones_loss
    print ("    mse loss: ", mseloss.item())
    print ("    inside_coverage_loss loss: ", inside_coverage_loss.item())
    # print ("    overlap_loss loss: ", overlap_loss.item())
    print ("    outside_loss loss: ", outside_loss.item())
    print ("    large_cones_loss loss: ", large_cones_loss.item())
    print ("    diversity_loss loss: ", diversity_loss.item())
    # print ("    pairwise_loss loss: ", pairwise_loss.item())

    # print()

    # radii = cone_params[..., 3].cpu().detach().numpy()
    # heights = cone_params[..., 4].cpu().detach().numpy()
    # print("     Radii range:", radii.min(), radii.max())
    # print("     Heights range:", heights.min(), heights.max())
    # print("     Radii std:", radii.std())
    # print("     Heights std:", heights.std())

    loss.backward(retain_graph=True)
    optimizer.step()
    print(f"Iteration {i}, Loss: {loss.item()}\n")
    return cone_params



def sample_primitive_batches(sphere_params, cone_params, num_spheres_sample, num_cones_sample, batch_size):
    """
    Sample a batch of random subsets of primitives.
    
    Args:
        sphere_params (torch.Tensor): Sphere parameters, shape [num_spheres, 4].
        cone_params (torch.Tensor): Cone parameters, shape [num_cones, 8].
        num_spheres_sample (int): Number of spheres to sample per subset.
        num_cones_sample (int): Number of cones to sample per subset.
        batch_size (int): Number of subsets in the batch.
    
    Returns:
        batched_sphere_params (torch.Tensor): Batched sphere parameters, shape [batch_size, num_spheres_sample, 4].
        batched_cone_params (torch.Tensor): Batched cone parameters, shape [batch_size, num_cones_sample, 8].
    """
    batched_sphere_params = []
    batched_cone_params = []

    for _ in range(batch_size):
        # Randomly sample indices for spheres and cones
        sphere_indices = random.sample(range(sphere_params.shape[0]), num_spheres_sample)
        cone_indices = random.sample(range(cone_params.shape[0]), num_cones_sample)

        # Append sampled primitives
        batched_sphere_params.append(sphere_params[sphere_indices])
        batched_cone_params.append(cone_params[cone_indices])

    # Stack for batch processing
    return torch.stack(batched_sphere_params), torch.stack(batched_cone_params)

def evaluate_batches(meta_model, batched_sphere_params, batched_cone_params, query_points, sdf_values):
    """
    Evaluate the fit of a batch of subsets of primitives.
    
    Args:
        meta_model (MetaNet): The MetaNet model.
        batched_sphere_params (torch.Tensor): Batched sphere parameters, shape [batch_size, num_spheres, 4].
        batched_cone_params (torch.Tensor): Batched cone parameters, shape [batch_size, num_cones, 8].
        query_points (torch.Tensor): Query points, shape [num_points, 3].
        sdf_values (torch.Tensor): Ground truth SDF values, shape [num_points].
    
    Returns:
        losses (torch.Tensor): Loss for each subset, shape [batch_size].
    """
    batch_size = batched_sphere_params.shape[0]
    losses = []

    for batch_idx in range(batch_size):
        sphere_params = batched_sphere_params[batch_idx]
        cone_params = batched_cone_params[batch_idx]

        # Forward pass for each subset
        combined_sdf, _, _ = meta_model(sphere_params, cone_params, query_points)
        loss = torch.mean((combined_sdf - sdf_values) ** 2)  # Compute reconstruction loss
        losses.append(loss)

    return torch.stack(losses)  # Return tensor of losses


def main():
    torch.autograd.set_detect_anomaly(True)

    num_cones = 64
    num_spheres = 64
    num_epochs = 1
    scoring_epochs = 2
    batch_size = 4
    num_spheres_sample = 10
    num_cones_sample = 10

    dataset_path = "./reference_models_processed"
    name = "dog"

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

    initial_cone_centers = initialize_cone_centers(points.cpu().numpy(), values, num_cones=num_cones, device=device)

    sphere_model = SphereNet(num_spheres=num_spheres).to(device)
    sphere_optimizer = torch.optim.Adam(sphere_model.parameters(), lr=0.0005)

    cone_model = ConeNet(num_cones=num_cones).to(device)
    cone_optimizer = torch.optim.Adam(cone_model.parameters(), lr=0.0005)

    meta_model = MetaNet(num_spheres, num_cones).to(device)
    meta_optimizer = torch.optim.Adam(meta_model.parameters(), lr=0.0005)

    for i in range(num_epochs):
        sphere_params = run_spherenet(sphere_optimizer, sphere_model, voxel_data, points, values, device, i)
        cone_params = run_conenet(cone_optimizer, cone_model, voxel_data, points, values, device, i, initial_cone_centers)
    
    sphere_params = sphere_params.cpu().detach()
    cone_params = cone_params.cpu().detach()

    for i in range(scoring_epochs):
        meta_optimizer.zero_grad()
        # combined_sdf, meta_sphere_params, meta_cone_params = meta_model(sphere_params, cone_params, points)

        # meta_loss = combined_fit_loss(combined_sdf, values)

        # meta_loss.backward()
        # meta_optimizer.step()

        batched_sphere_params, batched_cone_params = sample_primitive_batches(
            sphere_params, cone_params, num_spheres_sample, num_cones_sample, batch_size
        )

        batch_losses = evaluate_batches(meta_model, batched_sphere_params, batched_cone_params, points, values)

        best_idx = torch.argmin(batch_losses)
        best_sphere_params = batched_sphere_params[best_idx]
        best_cone_params = batched_cone_params[best_idx]

        # Forward pass with the best subset
        combined_sdf, sphere_weights, cone_weights = meta_model(best_sphere_params, best_cone_params, points)

        # Compute total loss with sparsity penalty
        sparsity_weight = 0.01
        meta_loss = torch.mean((combined_sdf - values) ** 2) + \
                    sparsity_weight * (torch.sum(torch.abs(sphere_weights)) + torch.sum(torch.abs(cone_weights)))
        meta_loss.backward()
        meta_optimizer.step()



        # print(f"Scoring Iteration {i}, Loss: {meta_loss.item()}")
        print(f"Scoring Iteration {i}, Best Batch Loss: {batch_losses[best_idx].item()}, Meta Loss: {meta_loss.item()}")


    torch.save(meta_model.state_dict(), "./output/meta_model.pth")

    meta_sphere_params = sphere_params.cpu().detach()
    meta_cone_params = cone_params.cpu().detach()

    visualize_primitives(meta_sphere_params, meta_cone_params, save_path="./output_primitives.obj")

    output_dir = "./output"
    os.makedirs(output_dir, exist_ok=True)
    
    np.save(os.path.join(output_dir, f"{name}_sphere_params.npy"), sphere_params.cpu().detach().numpy())
    np.save(os.path.join(output_dir, f"{name}_cone_params.npy"), cone_params.cpu().detach().numpy())

    visualise_spheres(points, values, sphere_params, reference_model=None, save_path=os.path.join(output_dir, f"{name}_spheres.obj"))
    visualize_cones(points, values, cone_params, save_path=os.path.join(output_dir, f"{name}_cones.obj"))

    torch.save(sphere_model.state_dict(), os.path.join(output_dir, f"{name}_sphere_model.pth"))
    torch.save(cone_model.state_dict(), os.path.join(output_dir, f"{name}_cone_model.pth"))

if __name__ == "__main__":
    main()