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
from pytorch3d.structures import Volumes
from pytorch3d.renderer import (
    FoVPerspectiveCameras, 
    VolumeRenderer,
    NDCMultinomialRaysampler,
    EmissionAbsorptionRaymarcher
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

# device = torch.device("cpu")
print("device", device)

class VolumeModel(torch.nn.Module):
    def __init__(self, renderer, volume_size=[64] * 3, voxel_size=0.1):
        super().__init__()
        # After evaluating torch.sigmoid(self.log_colors), we get 
        # densities close to zero.
        self.log_densities = torch.nn.Parameter(-4.0 * torch.ones(1, *volume_size))
        # After evaluating torch.sigmoid(self.log_colors), we get 
        # a neutral gray color everywhere.
        self.log_colors = torch.nn.Parameter(torch.zeros(3, *volume_size))
        self._voxel_size = voxel_size
        # Store the renderer module as well.
        self._renderer = renderer
        
    def set_renderer(self, renderer):
        self._renderer = renderer

    def forward(self, cameras):
        batch_size = cameras.R.shape[0]

        # Convert the log-space values to the densities/colors
        densities = torch.sigmoid(self.log_densities)
        colors = torch.sigmoid(self.log_colors)
        
        # Instantiate the Volumes object, making sure
        # the densities and colors are correctly
        # expanded batch_size-times.
        volumes = Volumes(
            densities = densities[None].expand(
                batch_size, *self.log_densities.shape),
            features = colors[None].expand(
                batch_size, *self.log_colors.shape),
            voxel_size=self._voxel_size, 
            volume_translation=[0,0,-2], # for display the whole volume
        )
        
        # Given cameras and volumes, run the renderer
        # and return only the first output value 
        # (the 2nd output is a representation of the sampled
        # rays which can be omitted for our purpose).
        return self._renderer(cameras=cameras, volumes=volumes)[0]
    

def get_volume(volume_ckpt):
    volume_size = 64
    volume_extent_world = 3.0
    raysampler = NDCMultinomialRaysampler(
        image_width=256,
        image_height=256,
        n_pts_per_ray=64,
        min_depth=0.1,
        max_depth=volume_extent_world,
    )
    raymarcher = EmissionAbsorptionRaymarcher()
    # Finally, instantiate the volumetric render
    # with the raysampler and raymarcher objects.
    renderer = VolumeRenderer(
        raysampler=raysampler, raymarcher=raymarcher,
    )
    
    volume_model = VolumeModel(
        renderer,
        volume_size=[volume_size] * 3, 
        voxel_size = volume_extent_world / volume_size,
    ).to(device)
    volume_model.load_state_dict(torch.load(volume_ckpt))
    return volume_model


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



class Pytorch3DVolumeActor(object):
    def __init__(self, single_pass=False):
        self.volume_model = get_volume('data/volume/volume_cow_64.pt')

    def init_gl(self,  height, width):
        self.width = width
        self.height = height

        
    def get_imdata(self, cameras, height=None, width=None):
        # images_raw = self.renderer(self.mesh).permute(0,3,1,2)
        with torch.no_grad():
            images_raw = self.volume_model(cameras)
            # upsample
            if height is not None and width is not None:
                images = F.interpolate(images_raw.permute(0,3,1,2), size=(height, width)).permute(0,2,3,1)
            else:
                images = images_raw
            im_np = (images.squeeze(0) * 255).to(torch.uint8).cpu().numpy()
        return im_np
   

    def display_gl(self, hmd_pose, eye2hmd, projection, texture_id):
        cameras = create_camera_from_pose(hmd_pose.reshape(1,3,4), eye2hmd.reshape(1,3,4), projection.reshape(1,4,4))
        imdata = self.get_imdata(cameras)
        glBindTexture(GL_TEXTURE_2D, texture_id)
        glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA8, self.width, self.height, 0, GL_RGBA, GL_UNSIGNED_BYTE, imdata)

    def get_stereo_views_np(self, hmd_pose, eye_left, projection_left, eye_right, projection_right):
        t0 = time.time()
        hmd_pose = np.stack([hmd_pose,hmd_pose],axis=0)
        eye2hmd = np.stack([eye_left, eye_right],axis=0)
        projection = np.stack([projection_left, projection_right])
        cameras = create_camera_from_pose(hmd_pose, eye2hmd, projection)
        imdata = self.get_imdata(cameras) 
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
                        imdata = self.get_imdata(cameras)[...,:3].astype(np.uint8)
                        try:
                            conn.sendall(imdata)
                        except Exception as e:
                            print("Error", e)
                            break
                except ConnectionResetError:
                    pass

def main():
    parser = argparse.ArgumentParser(description='Pytorch3D volume actor argument parser')
    parser.add_argument('--port',
                         type=int,
                         help='port (required by RemoteActorClient)',
                         default=9999)
    args = parser.parse_args()

    actor = Pytorch3DVolumeActor()
    actor.init_gl(1800,1800) # fake image size
    actor.run_as_server(port=args.port)


if __name__ == '__main__':
    main()
