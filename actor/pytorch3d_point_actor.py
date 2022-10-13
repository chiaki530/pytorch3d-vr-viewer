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
from pytorch3d.structures import Pointclouds
from pytorch3d.renderer import (
    PointsRasterizationSettings,
    PointsRenderer,
    FoVPerspectiveCameras,
    PointsRasterizer,
    AlphaCompositor,
)
import time
import argparse
# Setup
if torch.cuda.is_available():
    device = torch.device("cuda:0")
    torch.cuda.set_device(device)
else:
    device = torch.device("cpu")

# device = torch.device("cpu")
from numpysocket import NumpySocket
import open3d as o3d

print("device", device)

def get_point_cloud(obj_filename,scale=1,N=1):
    pointcloud = np.load(obj_filename)
    verts = torch.Tensor(pointcloud['verts']).to(device)  
    rgb = torch.Tensor(pointcloud['rgb']).to(device)
    verts = verts * scale

    # translation
    verts[...,1] = -verts[...,1] # flip y
    verts[...,2] += 1
    
    # downsample
    downsample_scale = 10
    verts = verts[::downsample_scale]
    rgb = rgb[::downsample_scale]
    print("verts",verts.shape, verts.max(dim=0), verts.min(dim=0), "downsample", downsample_scale) 
    point_cloud = Pointclouds(points=[verts], features=[rgb])
    point_cloud = point_cloud.extend(N)
    return point_cloud


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

def get_point_renderer(cameras, raster_settings):
    # Create a points renderer by compositing points using an alpha compositor (nearer points
    # are weighted more heavily). See [1] for an explanation.
    rasterizer = PointsRasterizer(cameras=cameras, raster_settings=raster_settings)
    renderer = PointsRenderer(
        rasterizer=rasterizer,
        compositor=AlphaCompositor()
    )
    return renderer

class Pytorch3DPointActor(object):
    def __init__(self, single_pass=False):
        N = 2 if single_pass else 1
        self.point_cloud = get_point_cloud('data/pointcloud/pointcloud.npz', N=N)

    def init_gl(self,  height, width):
        self.width = width
        self.height = height
        self.raster_settings = PointsRasterizationSettings(
            image_size=(512, 512), 
            radius = 0.005,
            points_per_pixel = 8
        )

        self.has_saved_1 = True
        self.has_saved_2 = True
        
    def get_imdata(self,  height=None, width=None):
        with torch.no_grad():
            images_raw = self.renderer(self.point_cloud)
            # upsample
            if height is not None and width is not None:
                images = F.interpolate(images_raw.permute(0,3,1,2), size=(height, width)).permute(0,2,3,1)
            else:
                images = images_raw
            im_np = (images[0] * 255).to(torch.uint8).cpu().numpy()
        return im_np
   

    def display_gl(self, hmd_pose, eye2hmd, projection, texture_id):
        cameras = create_camera_from_pose(hmd_pose.reshape(1,3,4), eye2hmd.reshape(1,3,4), projection.reshape(1,4,4))
        self.renderer = get_point_renderer(cameras, self.raster_settings)
        imdata = self.get_imdata(self.height, self.width)
        glBindTexture(GL_TEXTURE_2D, texture_id)
        glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA8, self.width, self.height, 0, GL_RGBA, GL_UNSIGNED_BYTE, imdata)


    def get_stereo_views_np(self, hmd_pose, eye_left, projection_left, eye_right, projection_right):
        t0 = time.time()
        hmd_pose = np.stack([hmd_pose,hmd_pose],axis=0)
        eye2hmd = np.stack([eye_left, eye_right],axis=0)
        projection = np.stack([projection_left, projection_right])
        cameras = create_camera_from_pose(hmd_pose, eye2hmd, projection)
        self.renderer = get_point_renderer(cameras, self.raster_settings)
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
                        self.renderer = get_point_renderer(cameras, self.raster_settings)
                        imdata = self.get_imdata()[...,:3].astype(np.uint8)
                        try:
                            conn.sendall(imdata)
                        except Exception as e:
                            print("Error", e)
                            break
                except ConnectionResetError:
                    pass


def main():
    parser = argparse.ArgumentParser(description='Pytorch3D point actor argument parser')
    parser.add_argument('--port',
                         type=int,
                         help='port (required by RemoteActorClient)',
                         default=9999)
    args = parser.parse_args()

    actor = Pytorch3DPointActor()
    actor.init_gl(1800,1800) # fake image size
    actor.run_as_server(port=args.port)


if __name__ == '__main__':
    main()
   