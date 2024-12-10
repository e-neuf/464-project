import torch
import torch.nn as nn
import numpy as np
import os
from MetaNet import MetaNet
from PrimitiveNet import visualize_primitives  # Assuming visualization is modularized
from SDF import determine_sphere_sdf, determine_cone_sdf
from PrimitiveNet import visualize_cones_small
from ConeNet import visualize_cones
from spherenet import visualise_spheres
import trimesh



def test_metanet(threshold, baseline_points, baseline_values, sphere_file, cone_file, sdf_file, output_dir="./output"):
    
    # Load saved sphere and cone parameters
    sdf_data = np.load(sdf_file)

    sphere_params_from_file = torch.tensor(np.load(sphere_file), dtype=torch.float32)
    cone_params_from_file = torch.tensor(np.load(cone_file), dtype=torch.float32)

    points = torch.tensor(sdf_data["sdf_points"], dtype=torch.float32)
    sdf_values = torch.tensor(sdf_data["sdf_values"], dtype=torch.float32)

    print("points shape: ", points.shape)
    print("values shape: ", sdf_values.shape)

    print("Sphere Params Shape: ", sphere_params_from_file.shape)
    print("Cone Params Shape: ", cone_params_from_file.squeeze(0).shape)

    sphere_sdf = determine_sphere_sdf(points, sphere_params_from_file)
    cone_sdf = determine_cone_sdf(points, cone_params_from_file)
    # print("Sphere SDF Shape: ", sphere_sdf.shape)
    print("Cone SDF Shape: ", cone_sdf.squeeze(0).shape)

    
    sphere_errors = torch.mean((sphere_sdf - sdf_values.unsqueeze(1)) ** 2, dim=0)  # [num_spheres]
    cone_errors = torch.mean((cone_sdf.squeeze(0) - sdf_values.unsqueeze(1)) ** 2, dim=0)  # [num_cones]

    sphere_error_min = sphere_errors.min()
    sphere_error_max = sphere_errors.max()
    
    cone_error_min = cone_errors.min()
    cone_error_max = cone_errors.max()

    normalized_sphere_errors = (sphere_errors - sphere_error_min) / (sphere_error_max - sphere_error_min)
    normalized_cone_errors = (cone_errors - cone_error_min) / (cone_error_max - cone_error_min)

    sphere_paramms_with_errors = torch.cat((sphere_params_from_file, normalized_sphere_errors.unsqueeze(1)), dim=1)
    cone_paramms_with_errors = torch.cat((cone_params_from_file.squeeze(0), normalized_cone_errors.unsqueeze(1)), dim=1)

    print("Sphere Params with Errors Shape: ", sphere_paramms_with_errors.shape)
    print("Cone Params with Errors Shape: ", cone_paramms_with_errors.shape)




    sphere_mask = sphere_paramms_with_errors[:, -1] < threshold
    cone_mask = cone_paramms_with_errors[:, -1] < threshold

    pruned_sphere_params = sphere_paramms_with_errors[sphere_mask]
    pruned_cone_params = cone_paramms_with_errors[cone_mask]

    pruned_sphere_params = pruned_sphere_params[:, :-1]
    pruned_cone_params = pruned_cone_params[:, :-1]

    print("Pruned Sphere Params Shape: ", pruned_sphere_params.shape)
    print("Pruned Cone Params Shape: ", pruned_cone_params.shape)

    # visualise_spheres(points, sdf_values, sphere_params_from_file, reference_model=None, save_path=None)
    # visualise_spheres(points, sdf_values, pruned_sphere_params, reference_model=None, save_path=None)

    print("Cone Params Shape: ", cone_params_from_file.shape)

    # visualize_cones_small(points, sdf_values, cone_params_from_file, save_path=None)
    # visualize_cones_small(points, sdf_values, pruned_cone_params, save_path=None)


    visualize_primitives(baseline_points, baseline_values, pruned_sphere_params, pruned_cone_params, save_path="./output/pruned_primitives.npy")

if __name__ == "__main__":

    dataset_path = "./reference_models_processed"
    name = "sofa"

    # Example paths for input files
    sphere_file = f"./output/{name}_sphere_params.npy"
    cone_file = f"./output/{name}_cone_params.npy"
    sdf_file = f"./reference_models_processed/{name}/voxel_and_sdf.npz"

    data = np.load(os.path.join(dataset_path, name, "voxel_and_sdf.npz"))

    voxel_data = data["voxels"]
    points = data["sdf_points"]
    values = data["sdf_values"]

    test_metanet(0.49, points, values, sphere_file, cone_file, sdf_file)
