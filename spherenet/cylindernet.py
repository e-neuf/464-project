import torch
import numpy as np
import torch.nn as nn
import os
import trimesh
from scipy.ndimage import gaussian_filter
from skimage.transform import resize
from dgcnn import DGCNNFeat

num_shapes = 10

def bsmin(a, dim, k=22.0, keepdim=False):
    dmix = -torch.logsumexp(-k * a, dim=dim, keepdim=keepdim) / k
    return dmix
    
def determine_cylinder_sdf(query_points, cylinder_params):
    centers = cylinder_params[:,:3]
    axes = cylinder_params[:,3:6]
    radii = cylinder_params[:,6]
    heights = cylinder_params[:,7]

    num_query_points = query_points.shape[0]
    num_cylinders = cylinder_params.shape[0]
    dist_to_cyl = torch.empty((num_query_points, num_cylinders))

    # normalize the axis vector
    axes_normalized = torch.nn.functional.normalize(axes)
    
    for i in range(num_cylinders):
        # calcualte distance from query points to cylinder axis
        scalar_points = torch.linalg.vecdot(query_points - centers[i,:], axes_normalized[i,:])
        projection = scalar_points[:, np.newaxis] * axes_normalized[i,:]
        closest_points = centers[i,:] + projection
        dist_to_axis = torch.linalg.vector_norm(query_points - closest_points, dim=1)
        point_is_within_radius = torch.le(dist_to_axis, radii[i])
        #print("num of points within the radius: "+str(torch.count_nonzero(point_is_within_radius)))

        # calcuate the points on the top and the bottom of the cylinder axis
        interm = (heights[i]/2) / torch.linalg.vector_norm(axes[i,:])
        top = centers[i,:] + interm * axes[i,:] 
        bottom = centers[i,:] - interm * axes[i,:]

        # calculate distance from query points to cylinder top/bottom
        dist_to_top = torch.linalg.vector_norm(projection - top, dim=1)
        dist_to_bottom = torch.linalg.vector_norm(projection - bottom, dim=1)
        dist_to_height = torch.minimum(dist_to_top, dist_to_bottom)
        point_is_within_height = torch.le((dist_to_top + dist_to_bottom), heights[i])
        #print("num of points within the height: "+str(torch.count_nonzero(point_is_within_height)))

        # use the appropriate distance for each query point
        point_is_inside = point_is_within_height & point_is_within_radius
        #print("num of points inside: "+str(torch.count_nonzero(point_is_inside)))
        #point_is_within_one = point_is_within_height ^ point_is_within_radius
        point_is_within_none = ~(point_is_within_height | point_is_within_radius)
        #print("num of points within none: "+str(torch.count_nonzero(point_is_within_none)))

        dist_to_cyl[point_is_within_height,i] = dist_to_axis[point_is_within_height]
        dist_to_cyl[point_is_within_radius,i] = dist_to_height[point_is_within_radius]
        inside = -1 * torch.minimum(torch.abs(dist_to_axis - radii[i]), dist_to_height)
        dist_to_cyl[point_is_inside,i] = inside[point_is_inside]
        dist_to_cyl[point_is_within_none,i] = torch.sqrt(dist_to_axis[point_is_within_none]**2 + dist_to_height[point_is_within_none]**2)
    
    return dist_to_cyl

class Decoder(nn.Module):
    def __init__(self):
        super(Decoder, self).__init__()
        in_ch = 256
        feat_ch = 512
        out_ch = num_shapes * 8
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
        num_params = sum(p.numel() for p in self.parameters())
        print("[num parameters: {}]".format(num_params))

    def forward(self, z):
        out1 = self.net1(z)
        out2 = self.net2(out1)
        return out2

class CylinderNet(nn.Module):
    def __init__(self, num_cylinders=32):
        super(CylinderNet, self).__init__()
        self.num_cylinders = num_cylinders
        self.encoder = DGCNNFeat(global_feat=True)
        self.decoder = Decoder()
        self.feature_mapper = nn.Linear(512, num_cylinders * 8)

    def forward(self, voxel_data, query_points):
        # Pass the voxel data through the encoder
        features = self.encoder(voxel_data)
        cylinder_params = self.decoder(features)
        cylinder_params = self.feature_mapper(cylinder_params).view(self.num_cylinders, 8)

        cylinder_params = torch.sigmoid(cylinder_params.view(-1, 8))
        #sphere_adder = torch.tensor([-0.5, -0.5, -0.5, 0.1]).to(sphere_params.device)
        #sphere_multiplier = torch.tensor([1.0, 1.0, 1.0, 0.4]).to(sphere_params.device)
        cylinder_adder = torch.tensor([-0.5, -0.5, -0.5, -0.5, -0.5, -0.5, 0.1, 0.1]).to(cylinder_params.device)
        cylinder_multiplier = torch.tensor([1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.2, 0.2]).to(cylinder_params.device)
        cylinder_params = cylinder_params * cylinder_multiplier + cylinder_adder
        
        cylinder_sdf = determine_cylinder_sdf(query_points, cylinder_params)
        return cylinder_sdf, cylinder_params

def visualize_cylinders(cylinder_params, points, values, reference_model=None, save_path=None):
    cylinder_params = cylinder_params.squeeze(0).cpu().detach().numpy() # Handle any extra batch dimensions safely
    cylinder_centers = cylinder_params[:, :3]
    cylinder_axes = cylinder_params[:, 3:6]
    cylinder_radii = cylinder_params[:, 6]
    cylinder_heights = cylinder_params[:, 7]
    scene = trimesh.Scene()

    for i in range(cylinder_params.shape[0]):
        cyl = trimesh.creation.cylinder(cylinder_heights[i], cylinder_radii[i])

        # Normalize the axis vector
        axis = cylinder_axes[i]
        if np.linalg.norm(axis) < 1e-6:
            axis = np.array([0, 0, 1])  # Default orientation if invalid
        axis = axis / np.linalg.norm(axis)

        # Compute the rotation matrix to align the cone with the orientation vector
        z_axis = np.array([0, 0, 1])
        rotation_axis = np.cross(z_axis, axis)
        if np.linalg.norm(rotation_axis) < 1e-6:
            rotation_matrix = np.eye(4)  # No rotation needed
        else:
            rotation_angle = np.arccos(np.dot(z_axis, axis))
            rotation_matrix = trimesh.transformations.rotation_matrix(rotation_angle, rotation_axis)

        cyl.apply_transform(rotation_matrix)
        cyl.apply_translation(cylinder_centers[i])
        scene.add_geometry(cyl)
    
    inside_points = points[values < 0]
    if len(inside_points) > 0:
        inside_points = trimesh.points.PointCloud(inside_points)
        inside_points.colors = np.array([[0, 0, 255, 255]] * len(inside_points.vertices))  # Blue color for inside points
        scene.add_geometry(inside_points)
        
    if save_path is not None:
        scene.export(save_path)

    scene.show()

def visualise_sdf(scene, points, values):
    inside_points = points[values < 0]
    #outside_points = points[values > 0]
    if len(inside_points) > 0:
        inside_points = trimesh.points.PointCloud(inside_points)
        #outside_points = trimesh.points.PointCloud(outside_points)
        inside_points.colors = np.array([[0, 0, 255, 255]] * len(inside_points.vertices))  # Blue color for inside points
        #outside_points.colors = np.array([[0, 255, 255, 255]] * len(outside_points.vertices))  # Red color for outside points
        scene.add_geometry(inside_points)
        #scene.add_geometry([inside_points, outside_points])

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

# loss functions
def penalize_large_cylinders(cylinder_params):
    """
    Penalize cylinders with large radii to encourage fitting finer features.

    Args:
        cylinder_params (torch.Tensor): Cylinder parameters (centers, axes, radii, and height).

    Returns:
        torch.Tensor: Penalty for large radii and height.
    """
    cylinder_radii = cylinder_params[:,6]
    cylinder_height = cylinder_params[:,7]
    return torch.mean(cylinder_radii ** 2 + cylinder_height ** 2)


def main():
    dataset_path = "./reference_models_processed"
    name = "dog"

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    path = os.path.join(dataset_path, name, "voxel_and_sdf.npz")

    # Load the voxel data from the .npz file
    data = np.load(path)
    #print(data.files)  # Inspect the contents of the .npz file

    voxel_data = data["voxels"]
    #centroid = data["centroid"]
    #scale = data["scale"]

    # Preprocess the voxel data
    voxel_data = preprocess_voxel_data(voxel_data)
    voxel_data = torch.from_numpy(voxel_data).float().to(device)

    # Convert voxel data to mesh
    #reference_model = voxel_to_mesh(voxel_data)

    # Load other necessary data
    points = data["sdf_points"]
    values = data["sdf_values"]

    # visualise_sdf(points, values)
    # visualise_voxels(voxel_data)

    # Apply the same transformations to the points
    #points = (points - centroid) / scale

    points = torch.from_numpy(points).float().to(device)
    values = torch.from_numpy(values).float().to(device)

    model = CylinderNet(num_cylinders=num_shapes).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.0005)

    num_epochs = 50   #change parameter for number of itearations
    for i in range(num_epochs):
        optimizer.zero_grad()
        cylinder_sdf, cylinder_params = model(
            voxel_data.unsqueeze(0), points
        )
        
        cylinder_sdf = bsmin(cylinder_sdf, dim=-1).to(device)
        loss = nn.MSELoss()(cylinder_sdf, values)
        #loss = penalize_large_cylinders(cylinder_params)
        loss.backward()
        optimizer.step()
        print(f"Iteration {i}, Loss: {loss.item()}")

        if loss < 0.0028:   #modify as needed
            break
        
    output_dir = "./output"
    os.makedirs(output_dir, exist_ok=True)
    np.save(os.path.join(output_dir, f"{name}_cylinder_params.npy"), cylinder_params.cpu().detach().numpy())
    
    print(cylinder_params)
    visualize_cylinders(cylinder_params, points, values)

    torch.save(model.state_dict(), os.path.join(output_dir, f"{name}_model.pth"))

if __name__ == "__main__":
    main()