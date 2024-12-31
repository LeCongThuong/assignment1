"""
Sample code to render various representations.

Usage:
    python -m starter.render_generic --render point_cloud  # 5.1
    python -m starter.render_generic --render parametric  --num_samples 100  # 5.2
    python -m starter.render_generic --render implicit  # 5.3
"""
import argparse
import pickle

import matplotlib.pyplot as plt
import mcubes
import numpy as np
import pytorch3d
import torch
import imageio
from starter.utils import get_device, get_mesh_renderer, get_points_renderer, unproject_depth_image
from pytorch3d.renderer.cameras import look_at_view_transform


def load_rgbd_data(path="data/rgbd_data.pkl"):
    with open(path, "rb") as f:
        data = pickle.load(f)
    return data


def render_bridge(
    point_cloud_path="data/bridge_pointcloud.npz",
    image_size=256,
    background_color=(1, 1, 1),
    device=None,
):
    """
    Renders a point cloud.
    """
    if device is None:
        device = get_device()
    renderer = get_points_renderer(
        image_size=image_size, background_color=background_color
    )
    point_cloud = np.load(point_cloud_path)
    verts = torch.Tensor(point_cloud["verts"][::50]).to(device).unsqueeze(0)
    rgb = torch.Tensor(point_cloud["rgb"][::50]).to(device).unsqueeze(0)
    point_cloud = pytorch3d.structures.Pointclouds(points=verts, features=rgb)
    R, T = pytorch3d.renderer.look_at_view_transform(4, 10, 0)
    cameras = pytorch3d.renderer.FoVPerspectiveCameras(R=R, T=T, device=device)
    rend = renderer(point_cloud, cameras=cameras)
    rend = rend.cpu().numpy()[0, ..., :3]  # (B, H, W, 4) -> (H, W, 3)
    return rend


def render_pc(data_path="data/rgbd_data.pkl", image_size=256,   background_color=(1, 1, 1), device=None, output_gif='images/pc_vis.gif'):
    data = load_rgbd_data(data_path)
    if device is None:
        device = get_device()

    renderer = get_points_renderer(
        image_size=image_size, background_color=background_color
    )
    rgb1, mask1, depth1, rgb2, mask2, depth2, cameras1, cameras2 = torch.tensor(data['rgb1']).to(device=device), torch.tensor(data['mask1']), torch.tensor(data['depth1']), torch.tensor(data['rgb2']).to(device=device), torch.tensor(data['mask2']), torch.tensor(data['depth2']), data['cameras1'], data['cameras2']
    print(rgb1.shape, type(rgb1), mask1.shape, type(mask1))
    points_1, rgb_1 = unproject_depth_image(rgb1, mask1, depth1, cameras1)
    points_1 = points_1.to(device=device)
    rgb_1 = rgb_1.to(device=device)
    points_2, rgb_2 = unproject_depth_image(rgb2, mask2, depth2, cameras2)
    points_2 = points_2.to(device=device)
    rgb_2 = rgb_2.to(device=device)
    
    points = torch.cat([points_1, points_2], dim=0)
    points = torch.unsqueeze(points, 0)
    features = torch.cat([rgb_1, rgb_2], dim=0)
    features = torch.unsqueeze(features, 0)
    point_cloud = pytorch3d.structures.Pointclouds(points=points, features=features)

    elevation = 30.0
    azimuth_list = np.arange(-180, 180, 5)
    distance = 6.0

    images = []
    for azimuth in azimuth_list:
        R, T = look_at_view_transform(distance, elevation, azimuth, degrees=True)
        R = pytorch3d.transforms.euler_angles_to_matrix(torch.Tensor([0, 0, np.pi]), "XYZ") @ R
        cameras = pytorch3d.renderer.FoVPerspectiveCameras(
            R=R, T=T, fov=60, device=device
        )

        rend = renderer(point_cloud, cameras=cameras)
        rend = rend.cpu().numpy()[0, ..., :3]
        images.append((rend*255).astype(np.uint8))
    imageio.mimsave(output_gif, images, fps=15)

def render_sphere(image_size=256, num_samples=200, device=None):
    """
    Renders a sphere using parametric sampling. Samples num_samples ** 2 points.
    """

    if device is None:
        device = get_device()

    phi = torch.linspace(0, 2 * np.pi, num_samples)
    theta = torch.linspace(0, np.pi, num_samples)
    # Densely sample phi and theta on a grid
    Phi, Theta = torch.meshgrid(phi, theta)

    x = torch.sin(Theta) * torch.cos(Phi)
    y = torch.cos(Theta)
    z = torch.sin(Theta) * torch.sin(Phi)

    points = torch.stack((x.flatten(), y.flatten(), z.flatten()), dim=1)
    color = (points - points.min()) / (points.max() - points.min())

    sphere_point_cloud = pytorch3d.structures.Pointclouds(
        points=[points], features=[color],
    ).to(device)

    cameras = pytorch3d.renderer.FoVPerspectiveCameras(T=[[0, 0, 3]], device=device)
    renderer = get_points_renderer(image_size=image_size, device=device)
    rend = renderer(sphere_point_cloud, cameras=cameras)
    return rend[0, ..., :3].cpu().numpy()


def render_torus(image_size=256, num_samples=200, device=None, output_gif='images/torus_360.gif'):
    if device is None:
        device = get_device()
    renderer = get_points_renderer(image_size=image_size, device=device)

    phi = torch.linspace(0, 2 * np.pi, num_samples)
    theta = torch.linspace(0, 2 * np.pi, num_samples)

    Phi, Theta = torch.meshgrid(phi, theta)

    R = 1
    r = 0.5
    x = (R + r * torch.cos(Theta)) * torch.cos(Phi)
    y = (R + r * torch.cos(Theta)) * torch.sin(Phi)
    z = r * torch.sin(Theta)

    points = torch.stack((x.flatten(), y.flatten(), z.flatten()), dim=1)
    color = (points - points.min()) / (points.max() - points.min())

    torus_point_cloud = pytorch3d.structures.Pointclouds(
        points=[points], features=[color],
    ).to(device)

    elevation = 30.0
    azimuth_list = np.arange(-180, 180, 5)
    distance = 6.0

    images = []
    for azimuth in azimuth_list:
        R, T = look_at_view_transform(distance, elevation, azimuth, degrees=True)
        cameras = pytorch3d.renderer.FoVPerspectiveCameras(
            R=R, T=T, fov=60, device=device
        )

        rend = renderer(torus_point_cloud, cameras=cameras)
        rend = rend.cpu().numpy()[0, ..., :3]
        images.append((rend*255).astype(np.uint8))
    imageio.mimsave(output_gif, images, fps=15)


def render_sphere_mesh(image_size=256, voxel_size=64, device=None):
    if device is None:
        device = get_device()
    min_value = -1.1
    max_value = 1.1
    X, Y, Z = torch.meshgrid([torch.linspace(min_value, max_value, voxel_size)] * 3)
    voxels = X ** 2 + Y ** 2 + Z ** 2 - 1
    vertices, faces = mcubes.marching_cubes(mcubes.smooth(voxels), isovalue=0)
    vertices = torch.tensor(vertices).float()
    faces = torch.tensor(faces.astype(int))
    # Vertex coordinates are indexed by array position, so we need to
    # renormalize the coordinate system.
    vertices = (vertices / voxel_size) * (max_value - min_value) + min_value
    textures = (vertices - vertices.min()) / (vertices.max() - vertices.min())
    textures = pytorch3d.renderer.TexturesVertex(vertices.unsqueeze(0))

    mesh = pytorch3d.structures.Meshes([vertices], [faces], textures=textures).to(
        device
    )
    lights = pytorch3d.renderer.PointLights(location=[[0, 0.0, -4.0]], device=device,)
    renderer = get_mesh_renderer(image_size=image_size, device=device)
    R, T = pytorch3d.renderer.look_at_view_transform(dist=3, elev=0, azim=180)
    cameras = pytorch3d.renderer.FoVPerspectiveCameras(R=R, T=T, device=device)
    rend = renderer(mesh, cameras=cameras, lights=lights)
    return rend[0, ..., :3].detach().cpu().numpy().clip(0, 1)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--render",
        type=str,
        default="point_cloud",
        choices=["point_cloud", "parametric", "implicit"],
    )
    parser.add_argument("--output_path", type=str, default="images/bridge.jpg")
    parser.add_argument("--image_size", type=int, default=256)
    parser.add_argument("--num_samples", type=int, default=100)
    args = parser.parse_args()
    if args.render == "point_cloud":
        # image = render_bridge(image_size=args.image_size)
        render_pc(image_size=args.image_size)
    elif args.render == "parametric":
        # image = render_sphere(image_size=args.image_size, num_samples=args.num_samples)
        render_torus(image_size=args.image_size, num_samples=args.num_samples)
    elif args.render == "implicit":
        image = render_sphere_mesh(image_size=args.image_size)
    else:
        raise Exception("Did not understand {}".format(args.render))
    # plt.imsave(args.output_path, image)

