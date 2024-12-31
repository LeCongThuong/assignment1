"""
Sample code to render a cow.

Usage:
    python -m starter.render_mesh --image_size 256 --output_path images/cow_render.jpg
"""
import argparse

import matplotlib.pyplot as plt
import pytorch3d
import torch
import numpy as np
from pytorch3d.renderer.cameras import look_at_view_transform
from starter.utils import get_device, get_mesh_renderer, load_cow_mesh, create_tetrahedron_mesh
import imageio

from starter.utils import get_device, get_mesh_renderer, load_cow_mesh


def render_cow(
    cow_path="data/cow.obj", image_size=256, color=[0.7, 0.7, 1], device=None, retexture=True
):
    # The device tells us whether we are rendering with GPU or CPU. The rendering will
    # be *much* faster if you have a CUDA-enabled NVIDIA GPU. However, your code will
    # still run fine on a CPU.
    # The default is to run on CPU, so if you do not have a GPU, you do not need to
    # worry about specifying the device in all of these functions.
    if device is None:
        device = get_device()

    # Get the renderer.
    renderer = get_mesh_renderer(image_size=image_size)

    # Get the vertices, faces, and textures.
    vertices, faces = load_cow_mesh(cow_path)
    vertices = vertices.unsqueeze(0)  # (N_v, 3) -> (1, N_v, 3)
    faces = faces.unsqueeze(0)  # (N_f, 3) -> (1, N_f, 3)
    textures = torch.ones_like(vertices)  # (1, N_v, 3)
    textures = textures * torch.tensor(color)  # (1, N_v, 3)
    if retexture:
        color1 = [0, 0, 1]
        color2 = [1, 0, 0]
        z_min, z_max = vertices[:, :, 2].min(), vertices[:, :, 2].max()
        alpha = (vertices[:, :, 2:] - z_min) / (z_max - z_min)
        textures = torch.tensor(color1).view((1, 3)) * (1 - alpha) + torch.tensor(color2).view(1, 3) * alpha
        
    mesh = pytorch3d.structures.Meshes(
        verts=vertices,
        faces=faces,
        textures=pytorch3d.renderer.TexturesVertex(textures),
    )
    mesh = mesh.to(device)

    # Prepare the camera:
    cameras = pytorch3d.renderer.FoVPerspectiveCameras(
        R=torch.eye(3).unsqueeze(0), T=torch.tensor([[0, 0, 3]]), fov=60, device=device
    )

    # Place a point light in front of the cow.
    lights = pytorch3d.renderer.PointLights(location=[[0, 0, -3]], device=device)

    rend = renderer(mesh, cameras=cameras, lights=lights)
    rend = rend.cpu().numpy()[0, ..., :3]  # (B, H, W, 4) -> (H, W, 3)
    # The .cpu moves the tensor to GPU (if needed).
    return rend

def render_gif_cow(
    cow_path="data/cow.obj", image_size=256, color=[0.7, 0.7, 1], device=None, output_gif="images/cow_360.gif", retexture=True
):
    # The device tells us whether we are rendering with GPU or CPU. The rendering will
    # be *much* faster if you have a CUDA-enabled NVIDIA GPU. However, your code will
    # still run fine on a CPU.
    # The default is to run on CPU, so if you do not have a GPU, you do not need to
    # worry about specifying the device in all of these functions.
    if device is None:
        device = get_device()

    # Get the renderer.
    renderer = get_mesh_renderer(image_size=image_size)

    # Get the vertices, faces, and textures.
    vertices, faces = load_cow_mesh(cow_path)
    vertices = vertices.unsqueeze(0)  # (N_v, 3) -> (1, N_v, 3)
    faces = faces.unsqueeze(0)  # (N_f, 3) -> (1, N_f, 3)
    textures = torch.ones_like(vertices)  # (1, N_v, 3)
    textures = textures * torch.tensor(color)  # (1, N_v, 3)
    if retexture:
        color1 = [0, 0, 1]
        color2 = [1, 0, 0]
        z_min, z_max = vertices[:, :, 2].min(), vertices[:, :, 2].max()
        alpha = (vertices[:, :, 2:] - z_min) / (z_max - z_min)
        textures = torch.tensor(color1).view((1, 3)) * (1 - alpha) + torch.tensor(color2).view(1, 3) * alpha
    mesh = pytorch3d.structures.Meshes(
        verts=vertices,
        faces=faces,
        textures=pytorch3d.renderer.TexturesVertex(textures),
    )
    mesh = mesh.to(device)

    # Prepare the camera:
    elevation = 30.0
    azimuth_list = np.arange(0, 360, 10)
    distance = 3.0

    images = []
    for azimuth in azimuth_list:
        R, T = look_at_view_transform(distance, elevation, azimuth, degrees=True)
        cameras = pytorch3d.renderer.FoVPerspectiveCameras(
            R=R, T=T, fov=60, device=device
        )
        # Place a point light in front of the cow.
        lights = pytorch3d.renderer.PointLights(location=[[0, 0, -3]], device=device)

        rend = renderer(mesh, cameras=cameras, lights=lights)
        rend = rend.cpu().numpy()[0, ..., :3]
        images.append((rend*255).astype(np.uint8))
    imageio.mimsave(output_gif, images, fps=15)


def render_gif_tetrahedron(image_size=256, color=[0.7, 0.7, 1], device=None, output_gif="images/tetraheron_360.gif"):
    vertices, faces = create_tetrahedron_mesh()
    vertices = vertices.unsqueeze(0)  # (N_v, 3) -> (1, N_v, 3)
    faces = faces.unsqueeze(0)  # (N_f, 3) -> (1, N_f, 3)
    textures = torch.ones_like(vertices)  # (1, N_v, 3)
    textures = textures * torch.tensor(color)  # (1, N_v, 3)

    if device is None: 
        device = get_device()
    
      # Get the renderer.
    renderer = get_mesh_renderer(image_size=image_size)

    mesh = pytorch3d.structures.Meshes(
        verts=vertices,
        faces=faces,
        textures=pytorch3d.renderer.TexturesVertex(textures),
    )
    mesh = mesh.to(device)

    # Prepare the camera:
    elevation = 30.0
    azimuth_list = np.arange(0, 360, 10)
    distance = 3.0

    images = []
    for azimuth in azimuth_list:
        R, T = look_at_view_transform(distance, elevation, azimuth, degrees=True)
        cameras = pytorch3d.renderer.FoVPerspectiveCameras(
            R=R, T=T, fov=60, device=device
        )
        # Place a point light in front of the cow.
        lights = pytorch3d.renderer.PointLights(location=[[0, 0, -3]], device=device)

        rend = renderer(mesh, cameras=cameras, lights=lights)
        rend = rend.cpu().numpy()[0, ..., :3]
        images.append((rend*255).astype(np.uint8))
    imageio.mimsave(output_gif, images, fps=15)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--cow_path", type=str, default="data/cow.obj")
    parser.add_argument("--output_path", type=str, default="images/cow_color_render.jpg")
    parser.add_argument("--image_size", type=int, default=256)
    args = parser.parse_args()
    image = render_cow(cow_path=args.cow_path, image_size=args.image_size)
    render_gif_cow(cow_path=args.cow_path, image_size=args.image_size, output_gif="images/cow_color_360.gif")
    # render_gif_tetrahedron(image_size=args.image_size, output_gif="images/tetraheron_360.gif")
    plt.imsave(args.output_path, image)
