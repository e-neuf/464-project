import torch
import numpy as np
import os
import trimesh
from scipy.ndimage import gaussian_filter
from skimage.transform import resize
from Loss import calculate_coverage_loss, calculate_huber_loss, calculate_inside_coverage_loss, calculate_graded_outside_loss, calculate_size_diversity_loss, penalize_large_spheres
from ConeNet import ConeNet
from SphereNet import SphereNet

def bsmin(a, dim, k=22.0, keepdim=False):
    dmix = -torch.logsumexp(-k * a, dim=dim, keepdim=keepdim) / k
    return dmix

def visualise_spheres(points, values, sphere_params, reference_model=None, save_path=None):
    sphere_params = sphere_params.cpu().detach().numpy()
    sphere_centers = sphere_params[..., :3]
    sphere_radii = np.abs(sphere_params[..., 3])
    scene = trimesh.Scene()

    # Calculate the centroid of the sphere cluster
    centroid = sphere_centers.mean(axis=0)

    for center, radius in zip(sphere_centers, sphere_radii):
        sphere = trimesh.creation.icosphere(radius=radius, subdivisions=2)
        sphere.apply_translation(center)
        scene.add_geometry(sphere)

    inside_points = points[values < 0]
    inside_points = trimesh.points.PointCloud(inside_points)
    inside_points.colors = [0, 0, 255, 255]  # Blue color for inside points
    scene.add_geometry([inside_points])
        
    if save_path is not None:
        scene.export(save_path)
    scene.show()

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

def main():
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

    # Apply the same transformations to the points
    # points = (points - centroid) \ scale

    points = torch.from_numpy(points).float().to(device)
    values = torch.from_numpy(values).float().to(device)

    model = SphereNet(num_spheres=256).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.0005)

    num_epochs = 50   #change parameter for number of itearations
    for i in range(num_epochs):
        optimizer.zero_grad()
        sphere_sdf, sphere_params = model(
            voxel_data.unsqueeze(0), points
        )
        
        # bsmin approximates the minimum of tensor sphere_sdf along the last dimension,
        # in this case the number of spheres. The function effectively approximates the
        # minimum signed distance from each query point to all spheres. Once the minimum
        # is approximated it can be used for loss calculations which will be used to
        # minimize the difference between predicted and truth SDF’s during training.
        sphere_sdf = bsmin(sphere_sdf, dim=-1).to(device)

        # Determine the loss function to train the model, i.e. the mean squared error between gt sdf field and predicted sdf field.
        mseloss = torch.mean((sphere_sdf - values) ** 2)
        
        # Bonus: Design additional losses that helps to achieve a better result.
        # mseloss = 0

        coverage_loss = calculate_coverage_loss(points, sphere_params, points, values, sphere_sdf)

        # overlap_loss = calculate_overlap_loss(sphere_params)

        huber_loss = calculate_huber_loss(sphere_sdf, values, delta=0.3)

        inside_coverage_loss = calculate_inside_coverage_loss(points, values, sphere_params)

        outside_loss = calculate_graded_outside_loss(sphere_params, ((0,0,0), (64,64,64)), buffer=0.3, penalty_scale=2.0)

        diversity_loss = calculate_size_diversity_loss(sphere_params)

        large_sphere_penalty = penalize_large_spheres(sphere_params)

        print(f"    MSL loss: {mseloss.item()}")
        print(f"    Inside coverage loss: {inside_coverage_loss.item()}")
        print(f"    Outside loss: {outside_loss.item()}")
        print(f"    Diversity loss: {diversity_loss.item()}")
        print(f"    Large sphere penalty: {large_sphere_penalty.item()}")
        # print(f"Huber loss: {huber_loss.item()}")
        # print(f"Coverage loss: {coverage_loss.item()}")
        # print(f"Overlap loss: {overlap_loss.item()}")

        loss = mseloss + 0.5*inside_coverage_loss + 0.5*outside_loss + 0.3*diversity_loss + 0.4*large_sphere_penalty
        loss.backward()
        optimizer.step()
        print(f"Iteration {i}, Loss: {loss.item()}")

        # if loss < 0.0028:   #modify as needed
        #     break
        
    output_dir = "./output"
    os.makedirs(output_dir, exist_ok=True)
    
    np.save(os.path.join(output_dir, f"{name}_sphere_params.npy"), sphere_params.cpu().detach().numpy())
    
    print(sphere_params)

    visualise_spheres(points, values, sphere_params, reference_model=None, save_path=os.path.join(output_dir, f"{name}_spheres.obj"))

    torch.save(model.state_dict(), os.path.join(output_dir, f"{name}_model.pth"))

if __name__ == "__main__":
    main()