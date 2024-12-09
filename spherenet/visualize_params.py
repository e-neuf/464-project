import numpy as np
import trimesh

def visualize_cylinders(cylinder_params, points, values, reference_model=None, save_path=None):
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

def main():
    cylinder_params = np.load("./output/dog_cylinder_params_04.npy")
    print("params shape: "+str(cylinder_params.shape))
    data = np.load("./reference_models_processed/dog/voxel_and_sdf.npz")
    values = data["sdf_values"]
    points = data["sdf_points"]

    visualize_cylinders(cylinder_params, points, values)

if __name__ == "__main__":
    main()