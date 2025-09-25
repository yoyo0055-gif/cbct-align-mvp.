import nibabel as nib
import numpy as np
from skimage import measure
import open3d as o3d
from nibabel.affines import apply_affine

# 1) Load CBCT nifti and extract bone surface
nii = nib.load('cbct.nii.gz')     # Input CBCT file (NIfTI format)
vol = nii.get_fdata()
aff = nii.affine

# Threshold for bone surface extraction (adjust per scan)
threshold = 300
verts, faces, normals, values = measure.marching_cubes(vol, level=threshold)

# Convert voxel coords to world coords
verts_world = apply_affine(aff, verts)

mesh = o3d.geometry.TriangleMesh()
mesh.vertices = o3d.utility.Vector3dVector(verts_world)
mesh.triangles = o3d.utility.Vector3iVector(faces)
mesh.compute_vertex_normals()
o3d.io.write_triangle_mesh('cbct_mesh.ply', mesh)

# 2) Load intraoral STL
stl = o3d.io.read_triangle_mesh('intraoral.stl')
stl.compute_vertex_normals()

# Sample point clouds
pcd_target = mesh.sample_points_poisson_disk(20000)
pcd_source = stl.sample_points_poisson_disk(20000)

# Helper function for preprocessing
voxel_size = 1.0  # mm (tune this)
def preprocess(pcd, voxel):
    pcd_down = pcd.voxel_down_sample(voxel)
    pcd_down.estimate_normals(
        o3d.geometry.KDTreeSearchParamHybrid(radius=voxel*2, max_nn=30))
    radius = voxel*5
    fpfh = o3d.pipelines.registration.compute_fpfh_feature(
        pcd_down,
        o3d.geometry.KDTreeSearchParamHybrid(radius, max_nn=100))
    return pcd_down, fpfh

src_down, src_f = preprocess(pcd_source, voxel_size)
tgt_down, tgt_f = preprocess(pcd_target, voxel_size)

# 3) Global registration with RANSAC
distance_threshold = voxel_size * 1.5
result_ransac = o3d.pipelines.registration.registration_ransac_based_on_feature_matching(
    src_down, tgt_down, src_f, tgt_f, mutual_filter=True,
    max_correspondence_distance=distance_threshold,
    estimation_method=o3d.pipelines.registration.TransformationEstimationPointToPoint(False),
    ransac_n=4,
    checkers=[
        o3d.pipelines.registration.CorrespondenceCheckerBasedOnEdgeLength(0.9),
        o3d.pipelines.registration.CorrespondenceCheckerBasedOnDistance(distance_threshold)
    ],
    criteria=o3d.pipelines.registration.RANSACConvergenceCriteria(4000000, 500)
)

# 4) Refine alignment with ICP
reg_icp = o3d.pipelines.registration.registration_icp(
    pcd_source, pcd_target,
    max_correspondence_distance=voxel_size*0.8,
    init=result_ransac.transformation,
    estimation_method=o3d.pipelines.registration.TransformationEstimationPointToPoint()
)

print("ICP fitness:", reg_icp.fitness, "inlier_rmse:", reg_icp.inlier_rmse)
print("Transformation:\n", reg_icp.transformation)

# Apply and save aligned STL
stl.transform(reg_icp.transformation)
o3d.io.write_triangle_mesh('stl_aligned.ply', stl)
