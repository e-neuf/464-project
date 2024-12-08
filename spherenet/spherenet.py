
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
    #we kmeans to calculate because it provides a good initial guess for the centers by gathering points that are close together
    #works on voxels that are inside the object
    kmeans = KMeans(n_clusters=num_cones)   #calculating kmeans for initial centers 
    voxel_points = voxel_points[values < 0] #ensuring only using points inside the object
    kmeans.fit(voxel_points) #fitting the kmeans
    centers = kmeans.cluster_centers_ 
    return torch.tensor(centers, dtype=torch.float32).to(device) 

def bsmin(a, dim, k=22.0, keepdim=False):
    # soft min of tensor a along dimension dim
    # k is the smoothness parameter
    # keepdim is whether to keep the dimension of the input
    dmix = -torch.logsumexp(-k * a, dim=dim, keepdim=keepdim) / k
    return dmix


def determine_cone_sdf(query_points, cone_params):
    # Extract cone parameters
    cone_centers = cone_params[:, :, :3]  # Cone centers (x, y, z)
    cone_radii = cone_params[:, :, 3]     # Cone radii
    cone_heights = cone_params[:, :, 4]   # Cone heights
    cone_orientations = cone_params[:, :, 5:8]  # Cone orientations (x, y, z)
    cone_orientations = F.normalize(cone_orientations, dim=-1)  # Normalize orientations to unit vectors

    # Calculate vectors from query points to cone centers
    vectors_points_to_centers = query_points[:, None, :] - cone_centers[None, :, :, :]

    # Calculate the Euclidean distance from query points to cone centers in the xy plane
    distance_points_to_centers = torch.norm(vectors_points_to_centers[:, :, :, :2], dim=-1)

    # Calculate the SDF for the cones
    cone_sdf = torch.max(
        distance_points_to_centers - cone_radii * (1 - vectors_points_to_centers[:, :, :, 2] / cone_heights),
        torch.abs(vectors_points_to_centers[:, :, :, 2]) - cone_heights
    )
    
    return cone_sdf

# def determine_cone_sdf(query_points, cone_params):
#     # Extract cone parameters
#     cone_centers = cone_params[:, :, :3]  # Cone centers (x, y, z)
#     cone_radii = cone_params[:, :, 3]     # Base radii
#     cone_heights = cone_params[:, :, 4]   # Heights
#     cone_orientations = cone_params[:, :, 5:8]  # Cone orientations (unit vector)
#     cone_orientations = F.normalize(cone_orientations, dim=-1)  # Ensure orientations are unit vectors

#     # Vectors from query points to cone centers
#     vectors_points_to_centers = query_points[:, None, :] - cone_centers[None, :, :, :]

#     # Project vectors onto the cone's axis (orientation)
#     projections_onto_axis = torch.sum(vectors_points_to_centers * cone_orientations[None, :, :, :], dim=-1)

#     # Perpendicular distances to the axis
#     perpendicular_distances = torch.norm(
#         vectors_points_to_centers - projections_onto_axis[..., None] * cone_orientations[None, :, :, :],
#         dim=-1
#     )

#     # Compute cone surface SDF
#     normalized_height = projections_onto_axis / cone_heights
#     normalized_radius = perpendicular_distances / cone_radii

#     # Distance to the slanted surface
#     slanted_surface_sdf = torch.sqrt(normalized_height ** 2 + normalized_radius ** 2) - 1

#     # Distance to the cap and base planes
#     cap_sdf = projections_onto_axis - cone_heights
#     base_sdf = -projections_onto_axis

#     # Combine all components
#     cone_sdf = torch.max(torch.max(slanted_surface_sdf, base_sdf), cap_sdf)

#     return cone_sdf


class Decoder(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(Decoder, self).__init__()
        feat_ch = 256  
        print (in_ch, feat_ch, out_ch)

        self.net1 = nn.Sequential(
            nn.utils.weight_norm(nn.Linear(in_ch, feat_ch)),
            nn.ReLU(inplace=True),
            nn.utils.weight_norm(nn.Linear(feat_ch, feat_ch)),
            nn.ReLU(inplace=True),
            nn.utils.weight_norm(nn.Linear(feat_ch, feat_ch)),
            nn.ReLU(inplace=True),
            nn.utils.weight_norm(nn.Linear(feat_ch, feat_ch)),
            nn.ReLU(inplace=True),
        )
        self.net2 = nn.Sequential(
            nn.utils.weight_norm(nn.Linear(feat_ch, feat_ch)),
            nn.ReLU(inplace=True),
            nn.utils.weight_norm(nn.Linear(feat_ch, feat_ch)),
            nn.ReLU(inplace=True),
            nn.utils.weight_norm(nn.Linear(feat_ch, feat_ch)),
            nn.ReLU(inplace=True),
            nn.utils.weight_norm(nn.Linear(feat_ch, feat_ch)),
            nn.ReLU(inplace=True),
            nn.Linear(feat_ch, out_ch),
        )

        num_params = sum(p.numel() for p in self.parameters())  #total number of parameters in the model
        print("[num parameters: {}]".format(num_params))

    def forward(self, z):
        out1 = self.net1(z)
        out2 = self.net2(out1)
        return out2

class ConeNet(nn.Module):
    def __init__(self, num_cones=32):  # default 32 cones 
        super(ConeNet, self).__init__()
        self.num_cones = num_cones
        self.encoder = DGCNNFeat(global_feat=True) #initialize encoder with DGCNN
        self.decoder = Decoder(512, num_cones * 8)  #initialize decoder with 8 parameters for each cone
        # 8 parameters: center (3), radius (1), height (1), orientation (3)

    def forward(self, voxel_data, query_points, initial_centers):
        # Pass the voxel data through the encoder
        features = self.encoder(voxel_data) #extract features
        #features 5D tensor: batch_size, channels, depth, height, width
        features = features.view(features.size(0), -1)  # Flatten the features from 5D to 2D

        # Decode the features into cone parameters
        cone_params = self.decoder(features)
        cone_params = torch.sigmoid(cone_params.view(-1, self.num_cones, 8))
        #sigmoid: squashes the output to be between 0 and 1

        #adjusting cone parameters
        # cone_adder = torch.tensor([-0.5, -0.5, -0.5, 0.08, 0.08, -1.0, -1.0, -1.0]).to(cone_params.device)
        cone_multiplier = torch.tensor([1.0, 1.0, 1.0, 0.1, 0.1, 1.0, 1.0, 1.0]).to(cone_params.device)
        cone_params = cone_params * cone_multiplier 

        # Use the initial centers for the first training iteration
        if self.training:
            cone_params[:, :, :3] = initial_centers

        cone_sdf = determine_cone_sdf(query_points, cone_params)

        return cone_sdf, cone_params

def compute_pca(data):
    if data.shape[0] < 2:  # Check if there are fewer than 2 points
        return np.array([0, 0, 1])  # Default orientation if not enough points
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
    cone_params = cone_params.cpu().detach().numpy() #covert to array 

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

            radius = float(radius)
            height = float(height)

            # Extract points within the region of the cone
            mask = np.linalg.norm(points.cpu().detach().numpy() - center, axis=1) < radius
            region_points = points[mask].cpu().detach().numpy()
            
            if len(region_points) > 1:
                # Compute the principal direction using PCA if points are in region
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
                rotation_matrix = np.eye(4) #identity matrix 
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


def preprocess_voxel_data(voxel_data, target_shape=(64, 64, 64), sigma=1.0):
    # voxel_data = voxel_data / voxel_data.max() #normalize the data
    voxel_data = voxel_data.astype(float) #convert to float
    voxel_data = (voxel_data - voxel_data.min()) / (voxel_data.max() - voxel_data.min()) #normalize
    voxel_data = gaussian_filter(voxel_data, sigma=sigma) #smooth
    voxel_data = resize(voxel_data, target_shape, mode='constant', anti_aliasing=True) #resample to target shape
    return voxel_data

def visualise_voxels(voxel_data):
    # Convert tensor to NumPy array
    if isinstance(voxel_data, torch.Tensor):
        voxel_data = voxel_data.cpu().numpy()

    voxel_data = voxel_data.astype(bool) #make boolean array
    voxel_grid = trimesh.voxel.VoxelGrid(voxel_data) 
    mesh = voxel_grid.as_boxes() #convert grid to mesh

    scene = trimesh.Scene()
    scene.add_geometry(mesh)
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

def penalize_large_cones(cone_params):
    """
    Penalize spheres with large radii to encourage fitting finer features.

    Args:
        sphere_params (torch.Tensor): Sphere parameters (centers and radii).
        weight (float): Penalty weight for large spheres.

    Returns:
        torch.Tensor: Penalty for large radii.
    """
    cone_radii = cone_params[:, :, 3]
    cone_height = cone_params[:, :, 4]
    return torch.mean(cone_radii ** 2 + cone_height ** 2)  # Penalizes large radii

def cone_overlap_loss(cone_params):
    batch_size, num_cones, _ = cone_params.shape    
    cone_centers = cone_params[:, :, :3] #first 3 values are the center
    cone_radii = cone_params[:, :, 3] #4th value is the radius
    cone_heights = cone_params[:, :, 4] #5th value is the height
    
    overlap_loss = 0.0
    
    for i in range(num_cones):
        for j in range(i + 1, num_cones):
            center_i = cone_centers[:, i, :]
            center_j = cone_centers[:, j, :]

            radius_i = cone_radii[:, i]
            radius_j = cone_radii[:, j]
            
            # Calculate the euclidean distance between the centers of the cones
            distance = torch.norm(center_i - center_j, dim=-1)
            
            # Calculate the overlap between the cones
            #overalp if sume of radius is greater than distance between centers
            overlap = torch.max(torch.tensor(0.0).to(cone_params.device), radius_i + radius_j - distance)
            
            # Add the overlap to the total overlap loss
            overlap_loss += overlap.mean()
    
    return overlap_loss / (num_cones * (num_cones - 1) / 2) #normalize

def height_diversity_loss(cone_params):
    heights = cone_params[..., 4]  # Extract cone heights
    mean_height = torch.mean(heights)
    diversity_loss = torch.mean((heights - mean_height) ** 2)  # Variance of heights
    return diversity_loss  # Penalize low variance (encourage variety)

def apply_dynamic_scaling(cone_params, epoch, num_epochs):
    scale_factor = torch.linspace(0.1, 2.0, steps=num_epochs).to(cone_params.device)  # Gradual scaling
    current_scale = scale_factor[epoch]
    cone_params = cone_params.clone()  
    cone_params[..., 4] = cone_params[..., 4] * current_scale  # Safely scale cone heights
    return cone_params

def main():
    dataset_path = "./data"
    name = "hand"

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load the voxel data from the .npz file
    data = np.load(f"reference_models_processed/{name}/voxel_and_sdf.npz")
    print(data.files)  # Inspect the contents of the .npz file

    voxel_data = data["voxels"]
    # centroid = data["centroid"]
    # scale = data["scale"]

    voxel_data = preprocess_voxel_data(voxel_data)
    voxel_data = torch.from_numpy(voxel_data).float().to(device)

    # Load other necessary data
    points = data["sdf_points"]
    values = data["sdf_values"]
    # values = values.astype(np.float32)
    # values = values / np.abs(values).max()  # Normalize SDF values

    # visualise_sdf(points, values)
    # visualise_voxels(voxel_data)
    # points = (points - centroid) / scale

    points = torch.from_numpy(points).float().to(device)
    values = torch.from_numpy(values).float().to(device)

    num_cones = 256

    initial_centers = initialize_cone_centers(points.cpu().numpy(), values, num_cones, device=device)

    # model = SphereNet(num_spheres=256).to(device)
    model = ConeNet(num_cones).to(device)  # Adjusted num_cones to 256
    #to change number of cones: 
    #update num cones in main function 
    #update num cones in ConeNet class

    optimizer = torch.optim.Adam(model.parameters(), lr=0.0002)


    print("Voxel data range:", voxel_data.min().item(), voxel_data.max().item())
    print("Points range:", points.min().item(), points.max().item())
    print("SDF values range:", values.min().item(), values.max().item())
    # print("Initial cone parameters:", initial_centers)

    num_epochs = 15   #change parameter for number of itearations
    for i in range(num_epochs):
        optimizer.zero_grad()
        cone_sdf, cone_params = model(
            voxel_data.unsqueeze(0), points, initial_centers
        )


        print("Predicted SDF values:", cone_sdf)
        print("Ground truth SDF values:", values)
        
        # bsmin approximates the minimum of tensor sphere_sdf along the last dimension,
        # in this case the number of spheres. The function effectively approximates the
        # minimum signed distance from each query point to all spheres. Once the minimum
        # is approximated it can be used for loss calculations which will be used to
        # minimize the difference between predicted and truth SDF’s during training.
        # sphere_sdf = bsmin(sphere_sdf, dim=-1).to(device)

        cone_params = apply_dynamic_scaling(cone_params, i, num_epochs)

        cone_sdf = bsmin(cone_sdf, dim=-1).to(device)

        # Determine the loss function to train the model, i.e. the mean squared error between gt sdf field and predicted sdf field.
        mseloss = torch.mean((cone_sdf - values) ** 2)
        inside_coverage_loss = calculate_inside_cone_coverage_loss(points, values, cone_params)
        overlap_loss = cone_overlap_loss(cone_params)
        outside_loss = calculate_graded_outside_loss(cone_params, ((0,0,0), (64,64,64)), buffer=0.3, penalty_scale=2.0)
        large_cones_loss = penalize_large_cones(cone_params)
        height_loss = height_diversity_loss(cone_params)

        loss = mseloss + inside_coverage_loss + 0.5 * overlap_loss + outside_loss + 1 * large_cones_loss + 1 * height_loss
        print ("mse loss: ", mseloss.item())
        print ("inside_coverage_loss: ", inside_coverage_loss.item())
        print ("overlap_loss: ", overlap_loss.item())
        print ("outside_loss: ", outside_loss.item())
        print ("large_cones_loss: ", large_cones_loss.item())
        print ("height_loss : ", height_loss.item())

        loss.backward()
        optimizer.step()
        print(f"Iteration {i}, Loss: {loss.item()}\n")
        
    output_dir = "./output"
    os.makedirs(output_dir, exist_ok=True)
    np.save(os.path.join(output_dir, f"{name}_cone_params.npy"), cone_params.cpu().detach().numpy())
    # print(cone_params)
    visualize_cones(points, values, cone_params, save_path=os.path.join(output_dir, f"{name}_cones.obj"))
    torch.save(model.state_dict(), os.path.join(output_dir, f"{name}_model.pth"))

if __name__ == "__main__":
    main()