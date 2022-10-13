import os
from venv import create
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt

# Util function for loading point clouds|
import numpy as np
# import cupy
from OpenGL.GL import * 
# Data structures and functions for rendering
from pytorch3d.structures import Meshes
from pytorch3d.renderer import TexturesUV
from pytorch3d.io import load_obj
from pytorch3d.renderer import (
    RasterizationSettings,
    FoVPerspectiveCameras,
    FoVPerspectiveCameras,
    PointLights, 
    MeshRasterizer,
    MeshRenderer,
    SoftPhongShader
)
import time
import argparse
from numpysocket import NumpySocket

# Setup
if torch.cuda.is_available():
    device = torch.device("cuda:0")
    torch.cuda.set_device(device)
else:
    device = torch.device("cpu")
print("device", device)


def get_mesh(obj_filename, scale=1, N=1):
    verts, faces, aux = load_obj(obj_filename)

    # flip y
    verts[...,1] = -verts[...,1]
    
    tex_maps = aux.texture_images
    verts_uvs = aux.verts_uvs.to(device)  # (V, 2)
    faces_uvs = faces.textures_idx.to(device)  # (F, 3)
    image = list(tex_maps.values())[0].to(device)[None]
    tex = TexturesUV(
        verts_uvs=[verts_uvs], faces_uvs=[faces_uvs], maps=image
    )
    # Create a textures object
    # tex = Textures(verts_uvs=verts_uvs, faces_uvs=faces_uvs, maps=texture_image)

    # Initialise the mesh with textures
    mesh = Meshes(verts=[verts.to(device)], faces=[faces.verts_idx.to(device)], textures=tex)
    mesh = mesh.scale_verts(scale)
    mesh = mesh.extend(N)
    mesh = mesh.offset_verts_(torch.tensor([0,-0.5,1]).to(device))
    return mesh

def create_camera_from_pose(poses, eye2hmd, proj):
    K = torch.from_numpy(proj).float()
    R_eye = torch.from_numpy(eye2hmd[:,:3,:3]).float()
    T_eye = torch.from_numpy(eye2hmd[:,:3,3]).float()
    R_hmd = torch.from_numpy(poses[:,:3,:3]).float()
    T_hmd = torch.from_numpy(poses[:,:3,3]).float()
    R = R_eye @ R_hmd
    T = R_eye @ T_hmd.unsqueeze(-1) + T_eye.unsqueeze(-1)
    T = T.squeeze(-1)
    cameras = FoVPerspectiveCameras(
        R = R,
        T = T,
        K = K,
        # znear=0.1,
        # zfar=500,
        # fov=90,
        device=device
    )
    return cameras

def get_mesh_renderer(cameras, image_size):
    # Define the settings for rasterization and shading. Here we set the output image to be of size
    # 512x512. As we are rendering images for visualization purposes only we will set faces_per_pixel=1
    # and blur_radius=0.0. We also set bin_size and max_faces_per_bin to None which ensure that 
    # the faster coarse-to-fine rasterization method is used. Refer to rasterize_meshes.py for 
    # explanations of these parameters. Refer to docs/notes/renderer.md for an explanation of 
    # the difference between naive and coarse-to-fine rasterization. 
    raster_settings = RasterizationSettings(
        image_size=image_size, 
        blur_radius=0.0, 
        faces_per_pixel=1, 
    )

    # Place a point light in front of the object. As mentioned above, the front of the cow is facing the 
    # -z direction. 
    lights = PointLights(device=device, location=[[0.0, 0.0, -3.0]])

    # Create a Phong renderer by composing a rasterizer and a shader. The textured Phong shader will 
    # interpolate the texture uv coordinates for each vertex, sample from a texture image and 
    # apply the Phong lighting model
    renderer = MeshRenderer(
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
    return renderer


class Pytorch3DMeshActor(object):
    def __init__(self, single_pass=False):
        N = 2 if single_pass else 1
        self.mesh = get_mesh('data/mesh/cow.obj', scale=0.5, N=N)

    def init_gl(self,  height, width):
        self.width = width
        self.height = height
        self.raster_settings = RasterizationSettings(
            image_size=(512, 512), 
            blur_radius=0.0, 
            faces_per_pixel=1, 
        )
        
    def get_imdata(self, height=None, width=None):
        with torch.no_grad():
            images_raw = self.renderer(self.mesh)
            # upsample
            if height is not None and width is not None:
                images = F.interpolate(images_raw.permute(0,3,1,2), size=(height, width)).permute(0,2,3,1)
            else:
                images = images_raw
            im_np = (images.squeeze(0) * 255).to(torch.uint8).cpu().numpy()
        return im_np
   

    def update_camera(self, modelview, projection):
        world_to_eye = modelview.transpose()
        K = projection.transpose()
        if K[3,2] < 0:
            K[:,2] = -K[:,2]
        R = torch.tensor(world_to_eye[:3,:3]).float().unsqueeze(0)
        tvec = torch.tensor(world_to_eye[:3,3]).float().unsqueeze(0)
        K = torch.tensor(K).float().unsqueeze(0)
        cameras = FoVPerspectiveCameras(
            R=R,
            T=tvec,
            K=K,
            # znear=0.1,
            # zfar=500,
            # fov=90,
            device=device
        )

        # Place a point light in front of the object. As mentioned above, the front of the cow is facing the 
        # -z direction. 
        lights = PointLights(device=device, location=[[0.0, 0.0, -3.0]])

        # Create a Phong renderer by composing a rasterizer and a shader. The textured Phong shader will 
        # interpolate the texture uv coordinates for each vertex, sample from a texture image and 
        # apply the Phong lighting model
        self.renderer = MeshRenderer(
            rasterizer=MeshRasterizer(
                cameras=cameras, 
                raster_settings=self.raster_settings
            ),
            shader=SoftPhongShader(
                device=device, 
                cameras=cameras,
                lights=lights
            )
        )

    def display_gl(self, hmd_pose, eye2hmd, projection, texture_id):
        cameras = create_camera_from_pose(hmd_pose.reshape(1,3,4), eye2hmd.reshape(1,3,4), projection.reshape(1,4,4))
        self.renderer = get_mesh_renderer(cameras, (512,512))
        imdata = self.get_imdata(self.height, self.width)
        glBindTexture(GL_TEXTURE_2D, texture_id)
        glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA8, self.width, self.height, 0, GL_RGBA, GL_UNSIGNED_BYTE, imdata)


    def get_stereo_views_np(self, hmd_pose, eye_left, projection_left, eye_right, projection_right):
        t0 = time.time()
        hmd_pose = np.stack([hmd_pose,hmd_pose],axis=0)
        eye2hmd = np.stack([eye_left, eye_right],axis=0)
        projection = np.stack([projection_left, projection_right])
        cameras = create_camera_from_pose(hmd_pose, eye2hmd, projection)
        self.renderer = get_mesh_renderer(cameras, (512,512))
        imdata = self.get_imdata(self.height, self.width) 
        t3 = time.time()
        print("time", t3-t0)
        return imdata
        
    def dispose_gl(self):
        pass

    
    def run_as_server(self, port=9999):
        with NumpySocket() as s:
            s.bind(('', port))
            print("Run in server mode...", flush=True)
            while True:
                try:
                    s.listen()
                    conn, addr = s.accept()
                    print(f"connected: {addr}")
                    while conn:
                        # receive camera data
                        cam_data = conn.recv()
                        # print("server receieve pose", cam_data.shape)
                        hmd_pose = cam_data[0,:3,:4]
                        eye2hmd = cam_data[1,:3,:4]
                        projection = cam_data[2,:4,:4]
                        height = cam_data[3,0,0]
                        width = cam_data[3,0,1]
                        # render
                        cameras = create_camera_from_pose(hmd_pose.reshape(1,3,4), eye2hmd.reshape(1,3,4), projection.reshape(1,4,4))
                        self.renderer = get_mesh_renderer(cameras, (256,256))
                        imdata = self.get_imdata()[...,:3].astype(np.uint8)
                        try:
                            conn.sendall(imdata)
                        except Exception as e:
                            print("Error", e)
                            break
                        # print("server send frame", imdata.shape, flush=True)
                except ConnectionResetError:
                    pass


def main():
    parser = argparse.ArgumentParser(description='Pytorch3D mesh actor argument parser')
    parser.add_argument('--port',
                         type=int,
                         help='port (required by RemoteActorClient)',
                         default=9999)

    actor = Pytorch3DMeshActor()
    args = parser.parse_args()
    actor.init_gl(1800,1800) # fake image size
    actor.run_as_server(port=args.port)


if __name__ == '__main__':
    main()
