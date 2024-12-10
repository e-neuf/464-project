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