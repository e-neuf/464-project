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
        in_ch = 512
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

class SphereNet(nn.Module):
    def __init__(self, num_spheres=num_shapes):
        super(SphereNet, self).__init__()
        self.num_spheres = num_spheres
        self.encoder = DGCNNFeat(global_feat=True)
        self.decoder = Decoder()

    def forward(self, voxel_data, query_points):
        # Pass the voxel data through the encoder
        features = self.encoder(voxel_data)
        sphere_params = self.decoder(features)
        #print("shape after decoder:")
        #print(sphere_params.shape)
        sphere_params = torch.sigmoid(sphere_params.view(-1, 8))
        #print("shape after sigmoid view thingy:")
        #print(sphere_params.shape)
        #sphere_adder = torch.tensor([-0.5, -0.5, -0.5, 0.1]).to(sphere_params.device)
        #sphere_multiplier = torch.tensor([1.0, 1.0, 1.0, 0.4]).to(sphere_params.device)
        sphere_adder = torch.tensor([-0.5, -0.5, -0.5, 0.1, 0.1, 0.1, 0.1, 0.1]).to(sphere_params.device)
        sphere_multiplier = torch.tensor([1.0, 1.0, 1.0,1.0, 1.0, 1.0, 0.4, 0.4]).to(sphere_params.device)
        sphere_params = sphere_params * sphere_multiplier + sphere_adder
        sphere_sdf = determine_sphere_sdf(query_points, sphere_params)
        return sphere_sdf, sphere_params

def visualise_spheres(sphere_params, reference_model=None, save_path=None):
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

    if reference_model is not None:
        # Apply the translation to align the mesh with the sphere cluster centroid
        reference_model.apply_translation(centroid)
        # Set the opacity of the mesh to 50%
        reference_model.visual.face_colors = [0, 0, 255, 128]  # Blue color with 50% opacity
        scene.add_geometry(reference_model)
        
    if save_path is not None:
        scene.export(save_path)
    scene.show()

def visualise_sdf(points, values):
    inside_points = points[values < 0]
    outside_points = points[values > 0]
    inside_points = trimesh.points.PointCloud(inside_points)
    outside_points = trimesh.points.PointCloud(outside_points)
    inside_points.colors = [0, 0, 1, 1]  # Blue color for inside points
    outside_points.colors = [1, 0, 0, 1]  # Red color for outside points
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

# taken from task 2 of the assignment
def create_cylinder_mesh(center, direction, radius, height, color=[0, 1, 0]):
    """
    Create a cylinder mesh in trimesh centered at `center` and aligned to `direction`.

    Args:
        center (np.ndarray): The center point of the cylinder.
        direction (np.ndarray): A vector indicating the cylinder's orientation.
        radius (float): The radius of the cylinder.
        height (float): The height of the cylinder.
        color (list): RGB color of the cylinder.

    Returns:
        trimesh.Trimesh: A trimesh object representing the cylinder.
    """
    # Create a cylinder aligned with the Z-axis
    cylinder = trimesh.creation.cylinder(radius=radius, height=height, sections=32)

    # Normalize the direction vector
    direction = np.array(direction)
    direction /= np.linalg.norm(direction)

    # Compute the rotation matrix to align the cylinder's Z-axis with the given direction vector
    z_axis = np.array([0, 0, 1])  # The default axis of the cylinder
    rotation_matrix = trimesh.geometry.align_vectors(z_axis, direction)

    # Apply rotation to the cylinder
    cylinder.apply_transform(rotation_matrix)

    # Translate the cylinder to the desired center position
    cylinder.apply_translation(center)

    # Apply color to the cylinder mesh
    cylinder.visual.face_colors = np.array(color + [1.0]) * 255  # Color the mesh faces

    return cylinder



def main():
    dataset_path = "./reference_models_processed"
    name = "dog"

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    path = os.path.join(dataset_path, name, "voxel_and_sdf.npz")

    # Load the voxel data from the .npz file
    data = np.load(path)
    #print(data.files)  # Inspect the contents of the .npz file

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
    points = (points - centroid) / scale

    points = torch.from_numpy(points).float().to(device)
    values = torch.from_numpy(values).float().to(device)

    model = SphereNet(num_spheres=num_shapes).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.0005)

    num_epochs = 10   #change parameter for number of itearations
    for i in range(num_epochs):
        optimizer.zero_grad()
        cylinder_sdf, cylinder_params = model(
            voxel_data.unsqueeze(0), points
        )
        
        cylinder_sdf = bsmin(cylinder_sdf, dim=-1).to(device)
        mseloss = nn.MSELoss()(cylinder_sdf, values)
        loss = mseloss
        loss.backward()
        optimizer.step()
        print(f"Iteration {i}, Loss: {loss.item()}")

        if loss < 0.0028:   #modify as needed
            break
        
    output_dir = "./output"
    os.makedirs(output_dir, exist_ok=True)
    np.save(os.path.join(output_dir, f"{name}_cylinder_params.npy"), cylinder_params.cpu().detach().numpy())
    
    #print(cylinder_params)

    #visualise_spheres(sphere_params, reference_model=None, save_path=os.path.join(output_dir, f"{name}_spheres.obj"))

    # modified from task 2 of the assignment to visualize the cylinders as meshes

    cylinders = []
    for i in range(num_shapes):
        cylinder_center, cylinder_axis, cylinder_radius, cylinder_height = (
            cylinder_params[i][0:3].detach().numpy(),
            cylinder_params[i][3:6].detach().numpy(),
            cylinder_params[i][6].detach().numpy(),
            cylinder_params[i][7].detach().numpy() ) # cylinder params row has 8 things in it
        cylinders.append(create_cylinder_mesh(
            cylinder_center, cylinder_axis, cylinder_radius, height=cylinder_height ))
    scene = trimesh.Scene(cylinders)
    scene.show()

    torch.save(model.state_dict(), os.path.join(output_dir, f"{name}_model.pth"))

if __name__ == "__main__":
    main()