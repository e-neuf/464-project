
from sklearn.cluster import KMeans
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import os
import trimesh
from scipy.ndimage import gaussian_filter
from skimage.transform import resize
from dgcnn import DGCNNFeat


def initialize_cone_centers(voxel_points, values, num_cones, device):
    kmeans = KMeans(n_clusters=num_cones)

    voxel_points = voxel_points[values < 0]
    kmeans.fit(voxel_points)
    centers = kmeans.cluster_centers_
    return torch.tensor(centers, dtype=torch.float32).to(device)

def cone_overlap_loss(cone_params):
    batch_size, num_cones, _ = cone_params.shape
    cone_centers = cone_params[:, :, :3]
    cone_radii = cone_params[:, :, 3]
    cone_heights = cone_params[:, :, 4]
    
    overlap_loss = 0.0
    
    for i in range(num_cones):
        for j in range(i + 1, num_cones):
            center_i = cone_centers[:, i, :]
            center_j = cone_centers[:, j, :]
            radius_i = cone_radii[:, i]
            radius_j = cone_radii[:, j]
            height_i = cone_heights[:, i]
            height_j = cone_heights[:, j]
            
            # Calculate the distance between the centers of the cones
            distance = torch.norm(center_i - center_j, dim=-1)
            
            # Calculate the overlap between the cones
            overlap = torch.max(torch.tensor(0.0).to(cone_params.device), radius_i + radius_j - distance)
            
            # Add the overlap to the total overlap loss
            overlap_loss += overlap.mean()
    
    return overlap_loss / (num_cones * (num_cones - 1) / 2)

def bsmin(a, dim, k=22.0, keepdim=False):
    dmix = -torch.logsumexp(-k * a, dim=dim, keepdim=keepdim) / k
    return dmix

def determine_sphere_sdf(query_points, sphere_params):
    sphere_centers = sphere_params[:, :, :3]
    sphere_radii = sphere_params[:, :, 3]
    vectors_points_to_centers = query_points[:, None, :] - sphere_centers[None, :, :, :]
    distance_points_to_centers = torch.norm(vectors_points_to_centers, dim=-1)
    sphere_sdf = distance_points_to_centers - sphere_radii
    return sphere_sdf

def determine_cone_sdf(query_points, cone_params):
    cone_centers = cone_params[:, :, :3]
    cone_radii = cone_params[:, :, 3]
    cone_heights = cone_params[:, :, 4]
    cone_orientations = cone_params[:, :, 5:8]
    cone_orientations = F.normalize(cone_orientations, dim=-1)  # Ensure normalized orientations
    vectors_points_to_centers = query_points[:, None, :] - cone_centers[None, :, :, :]
    distance_points_to_centers = torch.norm(vectors_points_to_centers[:, :, :, :2], dim=-1)
    cone_sdf = torch.max(
        distance_points_to_centers - cone_radii * (1 - vectors_points_to_centers[:, :, :, 2] / cone_heights),
        torch.abs(vectors_points_to_centers[:, :, :, 2]) - cone_heights
    )
    return cone_sdf

class Decoder(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(Decoder, self).__init__()
        feat_ch = 256  # Increased feature channels
        print (in_ch, feat_ch, out_ch)
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
        #     nn.utils.weight_norm(nn.Linear(feat_ch, feat_ch)),
        #     nn.ReLU(inplace=True),
        #     nn.utils.weight_norm(nn.Linear(feat_ch, feat_ch)),
        #     nn.ReLU(inplace=True),
        #     nn.Linear(feat_ch, out_ch),
        )
        num_params = sum(p.numel() for p in self.parameters())
        print("[num parameters: {}]".format(num_params))

    def forward(self, z):
        out1 = self.net1(z)
        out2 = self.net2(out1)
        return out2

class SphereNet(nn.Module):
    def __init__(self, num_spheres=128):
        super(SphereNet, self).__init__()
        self.num_spheres = num_spheres
        self.encoder = DGCNNFeat(global_feat=True)
        self.decoder = Decoder(256, 512)

    def forward(self, voxel_data, query_points):
        # Pass the voxel data through the encoder
        features = self.encoder(voxel_data)
        sphere_params = self.decoder(features)
        sphere_params = torch.sigmoid(sphere_params.view(-1, 4))
        sphere_adder = torch.tensor([-0.5, -0.5, -0.5, 0.1]).to(sphere_params.device)
        sphere_multiplier = torch.tensor([1.0, 1.0, 1.0, 0.4]).to(sphere_params.device)
        sphere_params = sphere_params * sphere_multiplier + sphere_adder
        sphere_sdf = determine_sphere_sdf(query_points, sphere_params)
        return sphere_sdf, sphere_params
    

class ConeNet(nn.Module):
    def __init__(self, num_cones=32):  # Adjusted num_cones to 256
        super(ConeNet, self).__init__()
        self.num_cones = num_cones
        self.encoder = DGCNNFeat(global_feat=True)
        self.decoder = Decoder(512, num_cones * 8)  # 8 parameters: center (3), radius (1), height (1), orientation (3)

    def forward(self, voxel_data, query_points, initial_centers):
        # Pass the voxel data through the encoder
        features = self.encoder(voxel_data)
        features = features.view(features.size(0), -1)  # Flatten the features

        # Decode the features into cone parameters
        cone_params = self.decoder(features)
        cone_params = torch.sigmoid(cone_params.view(-1, self.num_cones, 8))

        cone_adder = torch.tensor([-0.5, -0.5, -0.5, 0.1, 0.1, -1.0, -1.0, -1.0]).to(cone_params.device)
        cone_multiplier = torch.tensor([1.0, 1.0, 1.0, 0.4, 0.4, 2.0, 2.0, 2.0]).to(cone_params.device)
        cone_params = cone_params * cone_multiplier + cone_adder

        # Use the initial centers for the first training iteration
        if self.training:
            cone_params[:, :, :3] = initial_centers

        cone_sdf = determine_cone_sdf(query_points, cone_params)

        return cone_sdf, cone_params

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

def compute_pca(data):
    # Center the data
    data_mean = np.mean(data, axis=0)
    centered_data = data - data_mean
    
    # Compute the covariance matrix
    covariance_matrix = np.cov(centered_data, rowvar=False)
    
    # Compute the eigenvalues and eigenvectors
    eigenvalues, eigenvectors = np.linalg.eigh(covariance_matrix)
    
    # Get the principal component (eigenvector with the largest eigenvalue)
    principal_component = eigenvectors[:, np.argmax(eigenvalues)]
    
    return principal_component

def visualize_cones(points, values, cone_params, save_path=None):
    cone_params = cone_params.cpu().detach().numpy()
    cone_centers = cone_params[..., :3]
    cone_radii = np.abs(cone_params[..., 3])
    cone_heights = np.abs(cone_params[..., 4])
    cone_orientations = cone_params[..., 5:8]
    scene = trimesh.Scene()

    for i in range(cone_centers.shape[0]):
        for j in range(cone_centers.shape[1]):
            center = cone_centers[i, j]
            radius = cone_radii[i, j]
            height = cone_heights[i, j]
            orientation = cone_orientations[i, j]

            # Ensure radius and height are scalar values
            radius = float(radius)
            height = float(height)

            # Extract points within the region of the cone
            mask = np.linalg.norm(points.cpu().detach().numpy() - center, axis=1) < radius
            region_points = points[mask].cpu().detach().numpy()
            
            if len(region_points) > 0:
                # Compute the principal direction using PCA
                orientation = compute_pca(region_points)
            else:
                # Default orientation along the z-axis if no points are found
                orientation = np.array([0, 0, 1])
            
            # Normalize the orientation vector
            orientation = orientation / np.linalg.norm(orientation)

            # Create the cone
            cone = trimesh.creation.cone(radius=radius, height=height)

            # Compute the rotation matrix to align the cone with the orientation vector
            z_axis = np.array([0, 0, 1])
            rotation_axis = np.cross(z_axis, orientation)
            if np.linalg.norm(rotation_axis) < 1e-6:
                rotation_matrix = np.eye(4)
            else:
                rotation_angle = np.arccos(np.dot(z_axis, orientation))
                rotation_matrix = trimesh.transformations.rotation_matrix(rotation_angle, rotation_axis)

            # Apply the rotation and translation to the cone
            cone.apply_transform(rotation_matrix)
            cone.apply_translation(center)
            scene.add_geometry(cone)

    inside_points = points[values < 0].cpu().detach().numpy()
    outside_points = points[values > 0].cpu().detach().numpy()
    if len(inside_points) > 0:
        inside_points = trimesh.points.PointCloud(inside_points)
        outside_points = trimesh.points.PointCloud(outside_points)
        inside_points.colors = np.array([[0, 0, 255, 255]] * len(inside_points.vertices))  # Blue color for inside points
        outside_points.colors = np.array([[0, 255, 255, 255]] * len(outside_points.vertices))  # Blue color for inside points
        scene.add_geometry([inside_points])

    if save_path is not None:
        scene.export(save_path)
    scene.show()
    return voxel_data


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

def calculate_inside_cone_coverage_loss(sdf_points, sdf_values, sphere_params):
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
    sphere_sdf = determine_cone_sdf(inside_points, sphere_params)
    
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

def main():
    dataset_path = "./data"
    name = "dog"

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load the voxel data from the .npz file
    data = np.load("reference_models_processed/dog/voxel_and_sdf.npz")
    print(data.files)  # Inspect the contents of the .npz file

    voxel_data = data["voxels"]
    centroid = data["centroid"]
    scale = data["scale"]

    # Preprocess the voxel data
    voxel_data = preprocess_voxel_data(voxel_data)
    voxel_data = torch.from_numpy(voxel_data).float().to(device)

    # Convert voxel data to mesh

    # Load other necessary data
    points = data["sdf_points"]
    values = data["sdf_values"]

    # visualise_sdf(points, values)

    # visualise_voxels(voxel_data)

    # Apply the same transformations to the points
    # points = (points - centroid) \ scale

    points = torch.from_numpy(points).float().to(device)

    values = torch.from_numpy(values).float().to(device)

    initial_centers = initialize_cone_centers(points.cpu().numpy(), values, num_cones=32, device=device)

    # model = SphereNet(num_spheres=256).to(device)
    model = ConeNet(num_cones=32).to(device)  # Adjusted num_cones to 256
    optimizer = torch.optim.Adam(model.parameters(), lr=0.0005)

    num_epochs =15   #change parameter for number of itearations
    patience = 20
    for i in range(num_epochs):
        optimizer.zero_grad()
        cone_sdf, cone_params = model(
            voxel_data.unsqueeze(0), points, initial_centers
        )
        # sphere_sdf, sphere_params = model(
        #     voxel_data.unsqueeze(0), points
        # )
        
        # bsmin approximates the minimum of tensor sphere_sdf along the last dimension,
        # in this case the number of spheres. The function effectively approximates the
        # minimum signed distance from each query point to all spheres. Once the minimum
        # is approximated it can be used for loss calculations which will be used to
        # minimize the difference between predicted and truth SDF’s during training.
        # sphere_sdf = bsmin(sphere_sdf, dim=-1).to(device)
        cone_sdf = bsmin(cone_sdf, dim=-1).to(device)

        # Determine the loss function to train the model, i.e. the mean squared error between gt sdf field and predicted sdf field.
        # mseloss = torch.mean((sphere_sdf - values) ** 2)
        mseloss = torch.mean((cone_sdf - values) ** 2)

        inside_coverage_loss = calculate_inside_cone_coverage_loss(points, values, cone_params)

        overlap_loss = cone_overlap_loss(cone_params)

        outside_loss = calculate_graded_outside_loss(cone_params, ((0,0,0), (64,64,64)), buffer=0.3, penalty_scale=2.0)

        
        # Bonus: Design additional losses that helps to achieve a better result.
        # mseloss = 0

        loss = mseloss + inside_coverage_loss + 0.5 * overlap_loss + outside_loss
        print ("mse loss: ", mseloss.item())
        print ("inside_coverage_loss loss: ", inside_coverage_loss.item())
        print ("overlap_loss loss: ", overlap_loss.item())
        print ("outside_loss loss: ", outside_loss.item())

        loss.backward()
        optimizer.step()
        print(f"Iteration {i}, Loss: {loss.item()}\n")

        # if loss < 0.0028:   #modify as needed
        #     break
        
    output_dir = "./output"
    os.makedirs(output_dir, exist_ok=True)
    
    # np.save(os.path.join(output_dir, f"{name}_sphere_params.npy"), sphere_params.cpu().detach().numpy())
    np.save(os.path.join(output_dir, f"{name}_cone_params.npy"), cone_params.cpu().detach().numpy())
    
    # print(sphere_params)
    print(cone_params)

    # visualise_spheres(points, values, sphere_params, save_path=os.path.join(output_dir, f"{name}_spheres.obj"))
    visualize_cones(points, values, cone_params, save_path=os.path.join(output_dir, f"{name}_cones.obj"))
    
    torch.save(model.state_dict(), os.path.join(output_dir, f"{name}_model.pth"))

if __name__ == "__main__":
    main()