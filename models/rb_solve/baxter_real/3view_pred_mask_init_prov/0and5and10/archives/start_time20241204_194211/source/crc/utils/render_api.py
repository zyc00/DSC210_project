import os
from typing import List

import matplotlib.pyplot as plt
import numpy as np
# import pyrender
import torch
import tqdm
import trimesh
from dl_ext.primitive import safe_zip
from dl_ext.timer import EvalTime
from dl_ext.vision_ext.datasets.kitti.structures import Calibration
from pytorch3d.renderer import (
    look_at_view_transform,
    PointLights,
    RasterizationSettings,
    MeshRasterizer,
    SoftPhongShader, TexturesVertex, PerspectiveCameras, get_world_to_view_transform,
)
from pytorch3d.renderer.mesh.renderer import MeshRendererWithFragments
from pytorch3d.structures import Meshes
from tqdm import trange

# from crc.data.datasets import KinectRobot
from crc.structures.nvdiffrast_renderer import NVDiffrastRenderer
from crc.structures.sapien_kin import SAPIENKinematicsModelStandalone
from crc.utils.pn_utils import ptp, to_array
# ## 2. Create a renderer
#
# A renderer in PyTorch3D is composed of a **rasterizer** and a **shader** which each have a number of subcomponents such as a **camera** (orthographic/perspective). Here we initialize some of these components and use default values for the rest.
#
# In this example we will first create a **renderer** which uses a **perspective camera**, a **point light** and applies **Phong shading**. Then we learn how to vary different components using the modular API.
from crc.utils.utils_3d import canonical_to_camera_np, matrix_3x4_to_4x4, rotx, \
    roty_np, rotx_np, create_center_radius, transform_points
from crc.utils.vis3d_ext import Vis3D
import pytorch3d


# os.system('mkdir -p data/cow_mesh')
# os.system('wget -P data/cow_mesh https://dl.fbaipublicfiles.com/pytorch3d/data/cow_mesh/cow.obj')
# os.system('wget -P data/cow_mesh https://dl.fbaipublicfiles.com/pytorch3d/data/cow_mesh/cow.mtl')
# os.system('wget -P data/cow_mesh https://dl.fbaipublicfiles.com/pytorch3d/data/cow_mesh/cow_texture.png')
# Setup
# # Set paths
# DATA_DIR = "./data"
# obj_filename = os.path.join(DATA_DIR, "cow_mesh/cow.obj")
#
# # Load obj file
# mesh = load_objs_as_meshes([obj_filename], device=device)
#


def render_mesh_api(verts, faces, focal_length, dist=1.57, batch_size=20, image_size=512, textures=None):
    if isinstance(image_size, int):
        image_size = ((image_size, image_size),)
    if torch.cuda.is_available():
        device = torch.device("cuda:0")
        torch.cuda.set_device(device)
    else:
        device = torch.device("cpu")
    if textures is None:
        verts_rgb = torch.ones_like(verts)[None]  # (1, V, 3)
        textures = TexturesVertex(verts_rgb)

    mesh = Meshes(verts[None], faces[None], textures=textures).to(device=device)
    max_extent = ptp(mesh.get_bounding_boxes()[0], dim=1).max()

    # Get a batch of viewing angles.
    elev = torch.linspace(-89, 89, batch_size)
    azim = torch.linspace(-179, 179, batch_size)
    elev_interval = 180.0 / batch_size
    elev_offset = elev_interval * 0.4 * torch.rand_like(elev)
    elev = elev_offset + elev
    elev[0] = -89
    elev[-1] = 89
    azim_interval = 360.0 / batch_size
    azim_offset = azim_interval * 0.4 * torch.rand_like(azim)
    azim = azim_offset + azim
    azim[0] = -179
    azim[-1] = 179
    # All the cameras helper methods support mixed type inputs and broadcasting. So we can
    # view the camera from the same distance and specify dist=2.7 as a float,
    # and then specify elevation and azimuth angles for each viewpoint as tensors.
    # dist = 1.75
    R, T = look_at_view_transform(dist=dist * max_extent, elev=elev, azim=azim)
    wtvt = get_world_to_view_transform(R=R, T=T)
    # focal_length = 600.0
    cameras = PerspectiveCameras(focal_length=focal_length,
                                 principal_point=((image_size[0][0] / 2, image_size[0][1] / 2),),
                                 # K=K,
                                 R=R, T=T,
                                 device=device,
                                 in_ndc=False,
                                 image_size=image_size
                                 )

    raster_settings = RasterizationSettings(
        image_size=image_size[0],
        blur_radius=0.0,
        faces_per_pixel=1,
    )

    lights = PointLights(device=device, location=[[0.0, 0.0, -3.0]])

    renderer = MeshRendererWithFragments(
        rasterizer=MeshRasterizer(
            cameras=cameras,
            raster_settings=raster_settings
        ),
        shader=SoftPhongShader(
            device=device,
            cameras=cameras,
            lights=lights
        )
    )
    # todo: keep rasterizer only?

    meshes = mesh.extend(batch_size)
    # meshes = meshes.update_padded(wtvt.transform_points(verts).cuda())
    lights.location = torch.tensor([[0.0, 0.0, -3.0]], device=device)
    outputs = renderer(meshes, cameras=cameras, lights=lights)
    depths = outputs[1].zbuf[:, :, :, 0]
    images = outputs[0][..., :3]
    return images.cpu(), depths.cpu(), cameras, meshes


def pyrender_api(verts, faces, focal_length, dist=1.57, batch_size=20, image_size=512, textures=None,
                 round2=False, max_elev=89, min_elev=0, noise=True):
    import pyrender
    from pyrender import Node

    max_extent = max(verts.max(0).values - verts.min(0).values)
    elev = torch.linspace(-max_elev, max_elev, batch_size)
    elev_interval = 2 * max_elev / batch_size
    if noise:
        elev_offset = elev_interval * 0.4 * torch.rand_like(elev)
    else:
        elev_offset = 0
    elev = elev_offset + elev
    elev[0] = -max_elev
    elev[-1] = max_elev
    elev[(elev < min_elev) & (elev > 0)] = min_elev
    elev[(-min_elev < elev) & (elev < 0)] = -min_elev

    if not round2:
        azim = torch.linspace(-179, 179, batch_size)
        azim_interval = 360.0 / batch_size
        azim_offset = azim_interval * 0.4 * torch.rand_like(azim)
        azim = azim_offset + azim
        azim[0] = -179
        azim[-1] = 179
    else:
        azim1 = torch.linspace(-179, 179, batch_size // 2)
        azim2 = torch.linspace(181, 539, batch_size // 2)
        azim = torch.cat([azim1, azim2])
        azim_interval = 360.0 / batch_size
        azim_offset = azim_interval * 0.4 * torch.rand_like(azim)
        if not noise:
            azim_offset = 0
        azim = azim_offset + azim
        azim[0] = -179
        azim[-1] = 539

    gaussian_noise = torch.randn(batch_size) * 0.1
    dist = dist * max_extent + gaussian_noise
    R, T = look_at_view_transform(dist=dist, elev=elev, azim=azim)
    if isinstance(image_size, int):
        image_size = (image_size, image_size)
    cy, cx = image_size[0] / 2, image_size[1] / 2
    camera = pyrender.IntrinsicsCamera(fx=focal_length, fy=focal_length, cx=cx, cy=cy)
    coord_convert = np.eye(4)
    coord_convert[:3, :3] = roty_np(np.pi / 2) @ rotx_np(- np.pi / 2)
    coord_convert[:3, :3] = rotx_np(- np.pi)
    light = pyrender.SpotLight(color=np.ones(3), intensity=3.0,
                               innerConeAngle=np.pi / 16.0)
    scene = pyrender.Scene()
    scene.add(camera, pose=coord_convert)
    scene.add(light, pose=coord_convert)
    RT = torch.cat((R, T[..., None]), -1)
    poses = matrix_3x4_to_4x4(RT)
    r = pyrender.OffscreenRenderer(image_size[1], image_size[0])
    flags = pyrender.constants.RenderFlags.DEPTH_ONLY
    depths = []
    for pose in tqdm.tqdm(poses, leave=False):
        m = trimesh.Trimesh(canonical_to_camera_np(verts, pose=pose), faces)
        mesh = pyrender.Mesh.from_trimesh(m)
        node = Node(mesh=mesh)
        scene.add_node(node)
        depth = r.render(scene, flags)
        scene.remove_node(node)
        depths.append(depth)
    depths = torch.from_numpy(np.stack(depths))
    return depths, poses


def pyrender_ring_api(verts, faces, focal_length, dist=1.57, image_size=512, nelev=18, nazim=12, max_elev=80):
    import pyrender
    from pyrender import Node

    max_extent = max(verts.max(0).values - verts.min(0).values)

    elev = np.linspace(-max_elev, max_elev, nelev)
    all_RT = []
    for e in elev:
        RT = torch.from_numpy(create_center_radius(dist=dist, nrad=nazim, angle_z=e))
        all_RT.append(RT)
    all_RT = torch.cat(all_RT, 0)
    poses = matrix_3x4_to_4x4(all_RT)

    if isinstance(image_size, int):
        image_size = (image_size, image_size)
    cy, cx = image_size[0] / 2, image_size[1] / 2
    camera = pyrender.IntrinsicsCamera(fx=focal_length, fy=focal_length, cx=cx, cy=cy)
    coord_convert = np.eye(4)
    coord_convert[:3, :3] = rotx_np(-np.pi)
    light = pyrender.SpotLight(color=np.ones(3), intensity=3.0,
                               innerConeAngle=np.pi / 16.0)
    scene = pyrender.Scene()
    scene.add(camera, pose=coord_convert)
    scene.add(light, pose=coord_convert)
    r = pyrender.OffscreenRenderer(image_size[1], image_size[0])
    flags = pyrender.constants.RenderFlags.DEPTH_ONLY
    depths = []
    for pose in tqdm.tqdm(poses, leave=False, desc='rendering depth'):
        m = trimesh.Trimesh(canonical_to_camera_np(verts, pose=pose), faces)
        mesh = pyrender.Mesh.from_trimesh(m)
        node = Node(mesh=mesh)
        scene.add_node(node)
        depth = r.render(scene, flags)
        scene.remove_node(node)
        depths.append(depth)
    depths = torch.from_numpy(np.stack(depths))
    return depths, poses


@torch.no_grad()
def pytorch3d_render_ring_api(verts, faces, focal_length, cx=None, cy=None, dist=1.57, image_size=512, nelev=18,
                              nazim=12, max_elev=80, bin_size=None):
    # nelev, nazim = 18, 12
    # max_elev = 80

    elev = np.linspace(-max_elev, max_elev, nelev)
    all_RT = []
    for e in elev:
        RT = torch.from_numpy(create_center_radius(dist=dist, nrad=nazim, angle_z=e))
        all_RT.append(RT)
    all_RT = torch.cat(all_RT, 0)
    poses = matrix_3x4_to_4x4(all_RT)

    # image_width, image_height = 1600, 1200
    if not isinstance(focal_length, (tuple, list)):
        focal_length = (focal_length, focal_length)
    if cx is None:
        cx = image_size[1] / 2
    if cy is None:
        cy = image_size[0] / 2
    # cy, cx = image_size[0] / 2, image_size[1] / 2
    principal_point = (cx, cy)
    device = 'cuda'
    # coord_convert = np.eye(4)
    # coord_convert[:3, :3] = rotx_np(-np.pi)
    cameras = PerspectiveCameras(focal_length=(focal_length,),
                                 principal_point=(principal_point,),
                                 device=device,
                                 image_size=((image_size[1], image_size[0]),),
                                 # R=torch.from_numpy(rotx_np(-np.pi)).float().cuda()
                                 )
    raster_settings = RasterizationSettings(
        image_size=(image_size[0], image_size[1]),
        blur_radius=0.0,
        faces_per_pixel=1,
        bin_size=bin_size,
        max_faces_per_bin=None
    )
    rasterizer = MeshRasterizer(
        cameras=cameras,
        raster_settings=raster_settings
    )
    # transform verts
    verts_hom = Calibration.cart_to_hom(verts)
    t_verts_hom = torch.bmm(verts_hom[None].repeat(poses.shape[0], 1, 1), poses.permute(0, 2, 1).float())
    t_verts = t_verts_hom[:, :, :3] / t_verts_hom[:, :, 3:]
    # t_verts[:, :2] *= -1
    t_verts[:, :, :2] *= -1
    # faces = faces[None].repeat(poses.shape[0], 1, 1).long().cuda()
    # make small batches
    sbsz = 20
    depths = []
    evaltime = EvalTime(disable=True)
    evaltime('')
    for i in trange(0, t_verts.shape[0], sbsz):
        mt_verts = t_verts[i: i + sbsz]
        mt_verts = mt_verts.cuda()
        mt_faces = faces[None].repeat(mt_verts.shape[0], 1, 1).long().cuda()
        mesh = Meshes(mt_verts, mt_faces)
        evaltime('prepare mesh')
        fragments = rasterizer(mesh)
        evaltime('ras')
        depth = fragments.zbuf[..., 0].detach()
        depths.append(depth)
        evaltime('append')
    depths = torch.cat(depths).cpu()
    return depths, poses


class PyrenderRenderMeshApiHelper:
    _renderer = None
    H, W = None, None

    # def __init__(self):
    #     self._renderer = None

    @staticmethod
    def get_renderer(H, W):
        import pyrender
        if PyrenderRenderMeshApiHelper._renderer is None or H != PyrenderRenderMeshApiHelper.H or W != PyrenderRenderMeshApiHelper.W:
            PyrenderRenderMeshApiHelper._renderer = pyrender.OffscreenRenderer(W, H)
        return PyrenderRenderMeshApiHelper._renderer


def pyrender_render_mesh_api(mesh: trimesh.Trimesh, object_pose, H, W, K, return_depth=False):
    # os.environ["PYOPENGL_PLATFORM"] = "osmesa"
    os.environ["PYOPENGL_PLATFORM"] = "egl"
    import pyrender
    evaltime = EvalTime(disable=True)
    evaltime('')
    fu, fv, cu, cv = K[0, 0], K[1, 1], K[0, 2], K[1, 2]
    evaltime('1')
    camera = pyrender.IntrinsicsCamera(fx=fu, fy=fv, cx=cu, cy=cv)
    coord_convert = np.eye(4)
    coord_convert[:3, :3] = rotx(-np.pi)
    light = pyrender.SpotLight(color=np.ones(3), intensity=3.0,
                               innerConeAngle=np.pi / 16.0)
    object_pose = to_array(object_pose)
    vertices = transform_points(mesh.vertices, object_pose)
    mesh = trimesh.Trimesh(vertices, mesh.faces)
    evaltime('3')
    mesh = pyrender.Mesh.from_trimesh(mesh)
    evaltime('4')
    scene = pyrender.Scene()
    scene.add(mesh)
    scene.add(camera, pose=coord_convert)
    scene.add(light, pose=coord_convert)
    flags = pyrender.constants.RenderFlags.DEPTH_ONLY
    evaltime('5')
    r = PyrenderRenderMeshApiHelper.get_renderer(H, W)
    evaltime('6')
    depth = r.render(scene, flags)
    r.delete()
    evaltime('7')
    if return_depth:
        return depth
    mask = depth > 0
    evaltime('8')
    return mask


class NVdiffrastRenderMeshApiHelper:
    _renderer = None
    H, W = None, None

    @staticmethod
    def get_renderer(H, W):
        if NVdiffrastRenderMeshApiHelper._renderer is None or H != NVdiffrastRenderMeshApiHelper.H or W != NVdiffrastRenderMeshApiHelper.W:
            NVdiffrastRenderMeshApiHelper._renderer = NVDiffrastRenderer((H, W))
        return NVdiffrastRenderMeshApiHelper._renderer


def nvdiffrast_render_mesh_api(mesh: trimesh.Trimesh, object_pose, H, W, K, anti_aliasing=True):
    """
    :param mesh: trimesh mesh
    :param object_pose: object pose in camera coordinate
    :param H: image height
    :param W: image width
    :param K: camera intrinsics
    :return: mask
    """
    renderer = NVdiffrastRenderMeshApiHelper.get_renderer(H, W)
    verts = torch.from_numpy(mesh.vertices).float().cuda()
    faces = torch.from_numpy(mesh.faces).int().cuda()
    K = torch.from_numpy(K).float().cuda()
    object_pose = torch.from_numpy(object_pose).float().cuda()
    mask = renderer.render_mask(verts, faces, K, object_pose, anti_aliasing=anti_aliasing)
    mask = mask.cpu().numpy().astype(bool)
    return mask


def nvdiffrast_render_meshes_api(meshes: List[trimesh.Trimesh], object_poses, H, W, K, return_ndarray=True, return_sum=True):
    """
    :param meshes: list of trimesh mesh
    :param object_poses: list of object poses in camera coordinate
    :param H: image height
    :param W: image width
    :param K: camera intrinsics
    :return: mask np.array of shape (H, W), bool, 0 or 1
    """
    renderer = NVdiffrastRenderMeshApiHelper.get_renderer(H, W)
    masks = []
    K = torch.from_numpy(K).float().cuda()
    for mesh, object_pose in safe_zip(meshes, object_poses):
        verts = torch.from_numpy(mesh.vertices).float().cuda()
        faces = torch.from_numpy(mesh.faces).int().cuda()
        object_pose = torch.from_numpy(object_pose).float().cuda()
        mask = renderer.render_mask(verts, faces, K, object_pose, anti_aliasing=False)
        masks.append(mask)
    mask = torch.stack(masks)
    if return_sum:
        mask = mask.float().sum(0).clamp(max=1)
    if return_ndarray:
        mask = mask.cpu().numpy().astype(bool)
    return mask


def nvdiffrast_parallel_render_meshes_api(meshes: List[trimesh.Trimesh], object_poses, H, W, K, return_ndarray=True):
    """
    :param meshes: list of trimesh mesh
    :param object_poses: list of object poses in camera coordinate
    :param H: image height
    :param W: image width
    :param K: camera intrinsics
    :return: mask np.array of shape (H, W), bool, 0 or 1
    """
    renderer = NVdiffrastRenderMeshApiHelper.get_renderer(H, W)
    K = torch.from_numpy(K).float().cuda()
    object_poses = torch.from_numpy(np.stack(object_poses)).float().cuda()
    verts_list = []
    faces_list = []
    for mesh, object_pose in safe_zip(meshes, object_poses):
        verts = torch.from_numpy(mesh.vertices).float().cuda()
        faces = torch.from_numpy(mesh.faces).int().cuda()
        v = transform_points(verts, object_pose)
        verts_list.append(v)
        faces_list.append(faces)
    mesh = pytorch3d.structures.Meshes(verts=verts_list, faces=faces_list)
    verts, faces = mesh.verts_packed(), mesh.faces_packed().int()
    masks = renderer.batch_render_mask(verts, faces, K, anti_aliasing=False)
    mask = masks.float()
    if return_ndarray:
        mask = mask.cpu().numpy().astype(bool)
    return mask


class RenderXarmApiHelper:
    meshes = None
    sk = None
    _urdf_path = None

    @staticmethod
    def get_meshes():
        if RenderXarmApiHelper.meshes is None:
            RenderXarmApiHelper.meshes = {}
            from crc.structures.xarm_mapping import link_name_mesh_path_mapping
            for k, v in link_name_mesh_path_mapping.items():
                if v != "":
                    RenderXarmApiHelper.meshes[k] = trimesh.load_mesh(v)
        return RenderXarmApiHelper.meshes

    @staticmethod
    def get_sk(urdf_path):
        if RenderXarmApiHelper.sk is None or urdf_path != RenderXarmApiHelper._urdf_path:
            RenderXarmApiHelper.sk = SAPIENKinematicsModelStandalone(urdf_path)
            RenderXarmApiHelper._urdf_path = urdf_path
        return RenderXarmApiHelper.sk


def pyrender_render_xarm_api(urdf_path, Tc_c2b, qpos, H, W, K, link_indices=None, return_depth=False):
    """
    :param urdf_path: path to urdf
    :param Tc_c2b: Tcam_cam2base
    :param qpos: joint positions
    :param H: image height
    :param W: image width
    """
    xarm_meshes = RenderXarmApiHelper.get_meshes()
    sk = RenderXarmApiHelper.get_sk(urdf_path)
    names = [link.name for link in sk.robot.get_links()]
    poses = []
    meshes = []
    if link_indices is None:
        link_indices = range(8)
    for i in link_indices:
        pose = Tc_c2b @ sk.compute_forward_kinematics(np.asarray(qpos).tolist() + [0] * (sk.robot.dof - len(qpos)), i + 1).to_transformation_matrix()
        mesh = xarm_meshes[names[i + 1]]
        meshes.append(mesh)
        poses.append(pose)

    if return_depth is False:
        masks = []
        for mesh, pose in safe_zip(meshes, poses):
            mask = pyrender_render_mesh_api(mesh, pose, H, W, K)
            masks.append(mask)
        masks = np.stack(masks)
        mask = np.clip(np.sum(masks, axis=0), 0, 1)
        return mask
    else:
        depths = []
        for mesh, pose in safe_zip(meshes, poses):
            depth = pyrender_render_mesh_api(mesh, pose, H, W, K, return_depth=True)
            depth[depth == 0] = 100.0
            depths.append(depth)
        depths = np.stack(depths)
        depth = np.min(depths, axis=0)
        return depth


def pyrender_render_baxter_api(urdf_path, Tc_c2b, qpos, H, W, K, link_indices=None, return_depth=False):
    """
    :param urdf_path: path to urdf
    :param Tc_c2b: Tcam_cam2base
    :param qpos: joint positions
    :param H: image height
    :param W: image width
    """
    baxter_meshes = RenderBaxterApiHelper.get_meshes()
    sk = RenderBaxterApiHelper.get_sk(urdf_path)
    names = [link.name for link in sk.robot.get_links()]
    poses = []
    meshes = []
    qpos = np.asarray(qpos)
    assert qpos.shape[0] == 7
    if link_indices is None:
        link_ids = [32, 33, 34, 35, 36, 38]
    else:
        link_ids = link_indices
    # for i in link_indices:
    #     pose = Tc_c2b @ sk.compute_forward_kinematics(np.asarray(qpos).tolist() + [0] * (sk.robot.dof - len(qpos)), i + 1).to_transformation_matrix()
    #     mesh = xarm_meshes[names[i + 1]]
    #     meshes.append(mesh)
    #     poses.append(pose)
    for link in link_ids:
        pose = Tc_c2b @ sk.compute_forward_kinematics([0] * 8 + qpos.tolist(), link).to_transformation_matrix()
        meshes.append(baxter_meshes[names[link]])
        poses.append(pose)

    if return_depth is False:
        masks = []
        for mesh, pose in safe_zip(meshes, poses):
            mask = pyrender_render_mesh_api(mesh, pose, H, W, K)
            masks.append(mask)
        masks = np.stack(masks)
        mask = np.clip(np.sum(masks, axis=0), 0, 1)
        return mask
    else:
        depths = []
        for mesh, pose in safe_zip(meshes, poses):
            depth = pyrender_render_mesh_api(mesh, pose, H, W, K, return_depth=True)
            depth[depth == 0] = 100.0
            depths.append(depth)
        depths = np.stack(depths)
        depth = np.min(depths, axis=0)
        return depth


def nvdiffrast_render_xarm_api(urdf_path, Tc_c2b, qpos, H, W, K, return_ndarray=True, link_indices=None,
                               return_sum=True, Te_e2p=None):
    xarm_meshes = RenderXarmApiHelper.get_meshes()
    sk = RenderXarmApiHelper.get_sk(urdf_path)
    names = [link.name for link in sk.robot.get_links()]
    poses = []
    meshes = []
    if link_indices is None:
        link_indices = range(8)
    if Te_e2p is None:
        Te_e2p = np.eye(4)
    for i in link_indices:
        Tb_b2l = sk.compute_forward_kinematics(qpos, i + 1).to_transformation_matrix()
        if i > 7:  # magic number for xarm7
            Tb_b2e = sk.compute_forward_kinematics(qpos, 8).to_transformation_matrix()
            Te_e2l = np.linalg.inv(Tb_b2e) @ Tb_b2l
            Tb_b2l = Tb_b2e @ Te_e2p @ Te_e2l
        Tc_c2l = Tc_c2b @ Tb_b2l
        mesh = xarm_meshes[names[i + 1]]
        meshes.append(mesh)
        poses.append(Tc_c2l)
    mask = nvdiffrast_render_meshes_api(meshes, poses, H, W, K, return_ndarray=return_ndarray, return_sum=return_sum)
    return mask


class RenderMobileRobotApiHelper:
    meshes = None
    sk = None
    _urdf_path = None

    @staticmethod
    def get_meshes():
        if RenderMobileRobotApiHelper.meshes is None:
            RenderMobileRobotApiHelper.meshes = {}
            from crc.structures.mobile_robot_mapping import fixed_link_name_mesh_path_mapping
            for k, v in fixed_link_name_mesh_path_mapping.items():
                if v != "":
                    RenderMobileRobotApiHelper.meshes[k] = trimesh.load_mesh(v)
        return RenderMobileRobotApiHelper.meshes

    @staticmethod
    def get_sk(urdf_path):
        if RenderMobileRobotApiHelper.sk is None or urdf_path != RenderMobileRobotApiHelper._urdf_path:
            RenderMobileRobotApiHelper.sk = SAPIENKinematicsModelStandalone(urdf_path)
            RenderMobileRobotApiHelper._urdf_path = urdf_path
        return RenderMobileRobotApiHelper.sk


def nvdiffrast_render_mobilerobot_api(urdf_path, Tc_c2b, qpos, H, W, K, return_ndarray=True, link_indices=None,
                                      return_sum=True):
    xarm_meshes = RenderMobileRobotApiHelper.get_meshes()
    sk = RenderMobileRobotApiHelper.get_sk(urdf_path)
    names = [link.name for link in sk.robot.get_links()]
    poses = []
    meshes = []
    if link_indices is None:
        link_indices = range(7)
    # if Te_e2p is None:
    #     Te_e2p = np.eye(4)
    for i in link_indices:
        Tb_b2l = sk.compute_forward_kinematics(qpos, i).to_transformation_matrix()
        Tc_c2l = Tc_c2b @ Tb_b2l
        mesh = xarm_meshes[names[i]]
        meshes.append(mesh)
        poses.append(Tc_c2l)
    mask = nvdiffrast_render_meshes_api(meshes, poses, H, W, K, return_ndarray=return_ndarray, return_sum=return_sum)
    return mask


def nvdiffrast_parallel_render_xarm_api(urdf_path, robot_pose, qpos, H, W, K, return_ndarray=True):
    # evaltime = EvalTime()
    # evaltime("")
    xarm_meshes = RenderXarmApiHelper.get_meshes()
    sk = RenderXarmApiHelper.get_sk(urdf_path)
    names = [link.name for link in sk.robot.get_links()]
    poses = []
    meshes = []
    for i in range(8):
        pose = robot_pose @ sk.compute_forward_kinematics(np.asarray(qpos).tolist() + [0] * (sk.robot.dof - len(qpos)),
                                                          i + 1).to_transformation_matrix()
        mesh = xarm_meshes[names[i + 1]]
        meshes.append(mesh)
        poses.append(pose)
    # evaltime("prepare")
    mask = nvdiffrast_parallel_render_meshes_api(meshes, poses, H, W, K, return_ndarray=return_ndarray)
    # evaltime("render")
    return mask


class RenderBaxterApiHelper:
    meshes = None
    sk = None

    @staticmethod
    def get_meshes():
        if RenderBaxterApiHelper.meshes is None:
            RenderBaxterApiHelper.meshes = {}
            from crc.structures.baxter_mapping import link_name_mesh_path_mapping
            for k, v in link_name_mesh_path_mapping.items():
                if v != "":
                    RenderBaxterApiHelper.meshes[k] = trimesh.load_mesh(v)
        return RenderBaxterApiHelper.meshes

    @staticmethod
    def get_sk(urdf_path):
        if RenderBaxterApiHelper.sk is None:
            RenderBaxterApiHelper.sk = SAPIENKinematicsModelStandalone(urdf_path)
        return RenderBaxterApiHelper.sk


def nvdiffrast_render_baxter_api(urdf_path, robot_pose, qpos, H, W, K, return_ndarray=True, return_sum=True,
                                 link_indices=None):
    baxter_meshes = RenderBaxterApiHelper.get_meshes()
    sk = RenderBaxterApiHelper.get_sk(urdf_path)
    names = [link.name for link in sk.robot.get_links()]
    poses = []
    meshes = []
    qpos = np.asarray(qpos)
    assert qpos.shape[0] == 7
    if link_indices is None:
        link_ids = [32, 33, 34, 35, 36, 38]
    else:
        link_ids = link_indices
    for link in link_ids:
        pose = robot_pose @ sk.compute_forward_kinematics([0] * 8 + qpos.tolist(), link).to_transformation_matrix()
        meshes.append(baxter_meshes[names[link]])
        poses.append(pose)
    mask = nvdiffrast_render_meshes_api(meshes, poses, H, W, K, return_ndarray=return_ndarray, return_sum=return_sum)
    return mask


def get_ring_object_poses(min_dist, max_dist, min_elev=-80, max_elev=80, ndist=5, nelev=18, nazim=12,
                          trans_noise=0.0, endpoint=True, start_azim=0, end_azim=2 * np.pi, center=[0, 0, 0]):
    """
    :param min_dist:
    :param max_dist:
    :param min_elev:
    :param max_elev:
    :param ndist:
    :param nelev:
    :param nazim:
    :return: object poses in camera coordinate
    """
    elevs = np.linspace(min_elev, max_elev, nelev)
    dists = np.linspace(min_dist, max_dist, ndist)
    all_RT = []
    for d in dists:
        for e in elevs:
            RT = torch.from_numpy(create_center_radius(center=center, dist=d, nrad=nazim, angle_z=e, endpoint=endpoint,
                                                       start=start_azim, end=end_azim))
            all_RT.append(RT)
    all_RT = torch.cat(all_RT, 0)
    poses = matrix_3x4_to_4x4(all_RT)
    noise = torch.randn([poses.shape[0], 3]) * torch.tensor(trans_noise)
    poses[:, :3, 3] = poses[:, :3, 3] + noise
    return poses


class RenderRealmanApiHelper:
    meshes = None
    sk = None
    _urdf_path = None

    @staticmethod
    def get_meshes():
        if RenderRealmanApiHelper.meshes is None:
            RenderRealmanApiHelper.meshes = {}
            from crc.structures.realman_mapping import link_name_mesh_path_mapping
            for k, v in link_name_mesh_path_mapping.items():
                RenderRealmanApiHelper.meshes[k] = trimesh.load_mesh(v)
        return RenderRealmanApiHelper.meshes

    @staticmethod
    def get_sk(urdf_path):
        if RenderRealmanApiHelper.sk is None or urdf_path != RenderRealmanApiHelper._urdf_path:
            RenderRealmanApiHelper.sk = SAPIENKinematicsModelStandalone(urdf_path)
            RenderRealmanApiHelper._urdf_path = urdf_path
        return RenderRealmanApiHelper.sk


def nvdiffrast_render_realman_api(urdf_path, Tc_c2b, qpos, H, W, K, return_ndarray=True,
                                  link_indices=None,
                                  return_sum=True):
    realman_meshes = RenderRealmanApiHelper.get_meshes()
    sk = RenderRealmanApiHelper.get_sk(urdf_path)
    names = [link.name for link in sk.robot.get_links()]
    poses = []
    meshes = []
    if link_indices is None:
        link_indices = range(19)
    for i in link_indices:
        Tb_b2l = sk.compute_forward_kinematics(qpos, i).to_transformation_matrix()
        Tc_c2l = Tc_c2b @ Tb_b2l
        mesh = realman_meshes[names[i]]
        meshes.append(mesh)
        poses.append(Tc_c2l)
    mask = nvdiffrast_render_meshes_api(meshes, poses, H, W, K, return_ndarray=return_ndarray, return_sum=return_sum)
    return mask


def main():
    urdf_path = "data/xarm7_textured.urdf"
    robot_pose = np.array([[9.738e-01, 2.276e-01, 0.000e+00, 2.704e-17],
                           [7.784e-02, -3.330e-01, -9.397e-01, 3.372e-17],
                           [-2.139e-01, 9.150e-01, -3.420e-01, 8.000e-01],
                           [0.000e+00, 0.000e+00, 0.000e+00, 1.000e+00]])
    qpos = np.zeros(7)
    H, W = 720, 1280
    K = np.array([[9.068051757812500000e+02, 0.000000000000000000e+00, 6.501978759765625000e+02],
                  [0.000000000000000000e+00, 9.066802978515625000e+02, 3.677142944335937500e+02],
                  [0.000000000000000000e+00, 0.000000000000000000e+00, 1.000000000000000000e+00]])
    evaltime = EvalTime()
    evaltime("")
    for _ in tqdm.trange(1000):
        # mask = nvdiffrast_render_xarm_api(urdf_path, robot_pose, qpos, H, W, K, return_ndarray=False)
        mask = nvdiffrast_parallel_render_xarm_api(urdf_path, robot_pose, qpos, H, W, K, return_ndarray=False)
    evaltime("nvdiffrast_render_xarm_api")
    import matplotlib.pyplot as plt
    plt.imshow(mask.cpu())
    plt.show()


if __name__ == '__main__':
    main()
