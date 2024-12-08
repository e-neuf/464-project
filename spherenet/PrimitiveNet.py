import torch
import numpy as np
import os
import trimesh
from scipy.ndimage import gaussian_filter
from skimage.transform import resize
from Loss import calculate_coverage_loss, calculate_huber_loss, calculate_inside_coverage_loss, calculate_graded_outside_loss, calculate_size_diversity_loss, penalize_large_spheres, combined_coverage_loss, combined_overlap_loss
from ConeNet import ConeNet, visualize_cones
from SphereNet import SphereNet, visualise_spheres

def bsmin(a, dim, k=22.0, keepdim=False):
    dmix = -torch.logsumexp(-k * a, dim=dim, keepdim=keepdim) / k
    return dmix

def visualize_primitives(points, values, sphere_params, cone_params, save_path=None):
    """
    Visualize spheres and cones in the same scene.
    
    Args:
        points (torch.Tensor): The points to visualize (e.g., SDF query points).
        values (torch.Tensor): The SDF values for the points.
        sphere_params (torch.Tensor): Parameters for spheres (center [3], radius [1]).
        cone_params (torch.Tensor): Parameters for cones (apex [3], radius [1], height [1], orientation [3]).
        save_path (str, optional): Path to save the visualization. Defaults to None.
    """
    # Convert inputs to numpy
    sphere_params = sphere_params.cpu().detach().numpy()
    cone_params = cone_params.cpu().detach().numpy()
    
    # Extract sphere parameters
    sphere_centers = sphere_params[..., :3]
    sphere_radii = np.abs(sphere_params[..., 3])

    # Extract cone parameters
    cone_centers = cone_params[..., :3]
    cone_radii = np.abs(cone_params[..., 3])
    cone_heights = np.abs(cone_params[..., 4])
    cone_orientations = cone_params[..., 5:8]

    # Initialize a Trimesh scene
    scene = trimesh.Scene()

    # Add spheres to the scene
    for center, radius in zip(sphere_centers, sphere_radii):
        sphere = trimesh.creation.icosphere(radius=radius, subdivisions=2)
        sphere.apply_translation(center)
        scene.add_geometry(sphere)

    # Add cones to the scene
    for apex, radius, height, orientation in zip(cone_centers, cone_radii, cone_heights, cone_orientations):
        # Normalize orientation
        orientation = orientation / np.linalg.norm(orientation)
        
        # Create cone geometry
        cone = trimesh.creation.cone(radius=radius, height=height)

        # Align cone with orientation
        z_axis = np.array([0, 0, 1])
        rotation_axis = np.cross(z_axis, orientation)
        if np.linalg.norm(rotation_axis) > 1e-6:
            rotation_angle = np.arccos(np.dot(z_axis, orientation))
            rotation_matrix = trimesh.transformations.rotation_matrix(rotation_angle, rotation_axis)
            cone.apply_transform(rotation_matrix)

        # Translate cone to apex
        cone.apply_translation(apex)
        scene.add_geometry(cone)

    # Add inside points for context
    inside_points = points[values < 0].cpu().detach().numpy()
    if len(inside_points) > 0:
        inside_points = trimesh.points.PointCloud(inside_points)
        inside_points.colors = [0, 0, 255, 255]  # Blue for inside points
        scene.add_geometry(inside_points)

    # Save or display the scene
    if save_path:
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
        
        loss.backward()
        sphere_optimizer.step()
        print(f"Iteration {i}, Loss: {loss.item()}")

        return sphere_params

def run_conenet(cone_optimizer, cone_model, voxel_data, points, values, device, i):
    cone_optimizer.zero_grad()

    cone_sdf, cone_params = cone_model(
        voxel_data.unsqueeze(0), points
    )
    cone_sdf = bsmin(cone_sdf, dim=-1).to(device)

    cone_mseloss = torch.mean((cone_sdf - values) ** 2)
    
    print(f"    Cone MSL loss: {cone_mseloss.item()}")

    loss = (
        cone_mseloss
    )

    loss.backward()
    cone_optimizer.step()
    print(f"Iteration {i}, Loss: {loss.item()}")
    return cone_params

def main():
    torch.autograd.set_detect_anomaly(True)

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

    sphere_model = SphereNet(num_spheres=128).to(device)
    sphere_optimizer = torch.optim.Adam(sphere_model.parameters(), lr=0.0005)

    cone_model = ConeNet(num_cones=128).to(device)
    cone_optimizer = torch.optim.Adam(cone_model.parameters(), lr=0.0005)

    num_epochs = 20   #change parameter for number of itearations
    for i in range(num_epochs):
        # sphere_params = run_spherenet(sphere_optimizer, sphere_model, voxel_data, points, values, device, i)
        cone_params = run_conenet(cone_optimizer, cone_model, voxel_data, points, values, device, i)
        
    output_dir = "./output"
    os.makedirs(output_dir, exist_ok=True)
    
    # np.save(os.path.join(output_dir, f"{name}_sphere_params.npy"), sphere_params.cpu().detach().numpy())
    np.save(os.path.join(output_dir, f"{name}_cone_params.npy"), cone_params.cpu().detach().numpy())

    # print(sphere_params)

    # visualise_spheres(points, values, sphere_params, reference_model=None, save_path=os.path.join(output_dir, f"{name}_spheres.obj"))

    visualize_cones(points, values, cone_params, save_path=os.path.join(output_dir, f"{name}_cones.obj"))

    # visualize_primitives(points, values, sphere_params, cone_params, save_path="combined_primitives.obj")

    torch.save(sphere_model.state_dict(), os.path.join(output_dir, f"{name}_model.pth"))

if __name__ == "__main__":
    main()