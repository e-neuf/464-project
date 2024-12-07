import torch
import numpy as np
import torch.nn as nn
import os
import trimesh
from scipy.ndimage import gaussian_filter
from skimage.transform import resize
from dgcnn import DGCNNFeat

def bsmin(a, dim, k=22.0, keepdim=False):
    dmix = -torch.logsumexp(-k * a, dim=dim, keepdim=keepdim) / k
    return dmix

def determine_sphere_sdf(query_points, sphere_params):
    sphere_centers = sphere_params[:,:3]
    sphere_radii = sphere_params[:,3]
    vectors_points_to_centers = query_points[:, None, :] - sphere_centers[None, :, :]
    distance_points_to_centers = torch.norm(vectors_points_to_centers, dim=-1)
    sphere_sdf = distance_points_to_centers - sphere_radii
    return sphere_sdf

class Decoder(nn.Module):
    def __init__(self):
        super(Decoder, self).__init__()
        in_ch = 256
        feat_ch = 256
        out_ch = 512
        self.net1 = nn.Sequential(
            nn.utils.weight_norm(nn.Linear(in_ch, feat_ch)),
            nn.ReLU(inplace=True),
            nn.utils.weight_norm(nn.Linear(feat_ch, feat_ch)),
            nn.ReLU(inplace=True),
            # nn.utils.weight_norm(nn.Linear(feat_ch, feat_ch)),
            # nn.ReLU(inplace=True),
            # nn.utils.weight_norm(nn.Linear(feat_ch, feat_ch)),
            # nn.ReLU(inplace=True),
        )
        self.net2 = nn.Sequential(
            nn.utils.weight_norm(nn.Linear(feat_ch, feat_ch)),
            nn.ReLU(inplace=True),
            nn.utils.weight_norm(nn.Linear(feat_ch, feat_ch)),
            nn.ReLU(inplace=True),
            # nn.utils.weight_norm(nn.Linear(feat_ch, feat_ch)),
            # nn.ReLU(inplace=True),
            # nn.utils.weight_norm(nn.Linear(feat_ch, feat_ch)),
            # nn.ReLU(inplace=True),
            nn.Linear(feat_ch, out_ch),
        )
        num_params = sum(p.numel() for p in self.parameters())
        print("[num parameters: {}]".format(num_params))

    def forward(self, z):
        out1 = self.net1(z)
        out2 = self.net2(out1)
        return out2

class SphereNet(nn.Module):
    def __init__(self, num_spheres=512):
        super(SphereNet, self).__init__()
        self.num_spheres = num_spheres
        self.encoder = DGCNNFeat(global_feat=True)
        self.decoder = Decoder()

    def forward(self, voxel_data, query_points):
        # Pass the voxel data through the encoder
        features = self.encoder(voxel_data)
        sphere_params = self.decoder(features)
        sphere_params = torch.sigmoid(sphere_params.view(-1, 4))
        sphere_adder = torch.tensor([-0.5, -0.5, -0.5, 0.01]).to(sphere_params.device)
        sphere_multiplier = torch.tensor([1.0, 1.0, 1.0, 0.2]).to(sphere_params.device)
        sphere_params = sphere_params * sphere_multiplier + sphere_adder
        sphere_sdf = determine_sphere_sdf(query_points, sphere_params)
        return sphere_sdf, sphere_params

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

# def calculate_outside_loss(sphere_params, voxel_bounds):
#     """
#     Penalize spheres that extend outside the voxel volume.

#     Args:
#         sphere_params (torch.Tensor): Sphere parameters (centers and radii).
#         voxel_bounds (tuple): Min and max bounds of the voxel grid as ((xmin, ymin, zmin), (xmax, ymax, zmax)).

#     Returns:
#         torch.Tensor: The outside loss.
#     """
#     sphere_centers = sphere_params[:, :3]
#     sphere_radii = sphere_params[:, 3]
    
#     # Voxel bounds
#     (xmin, ymin, zmin), (xmax, ymax, zmax) = voxel_bounds

#     # Check if spheres exceed the bounds
#     outside_xmin = torch.relu(xmin - (sphere_centers[:, 0] - sphere_radii))
#     outside_ymin = torch.relu(ymin - (sphere_centers[:, 1] - sphere_radii))
#     outside_zmin = torch.relu(zmin - (sphere_centers[:, 2] - sphere_radii))

#     outside_xmax = torch.relu((sphere_centers[:, 0] + sphere_radii) - xmax)
#     outside_ymax = torch.relu((sphere_centers[:, 1] + sphere_radii) - ymax)
#     outside_zmax = torch.relu((sphere_centers[:, 2] + sphere_radii) - zmax)

#     # Sum up all penalties for spheres outside the voxel bounds
#     outside_loss = (
#         torch.mean(outside_xmin + outside_ymin + outside_zmin +
#                    outside_xmax + outside_ymax + outside_zmax)
#     )
    
#     return outside_loss

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