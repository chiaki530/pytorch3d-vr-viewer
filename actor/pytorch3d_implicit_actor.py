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
    NDCMultinomialRaysampler,
    EmissionAbsorptionRaymarcher,
    ImplicitRenderer,
    RayBundle,
    ray_bundle_to_ray_points,
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


class HarmonicEmbedding(torch.nn.Module):
    def __init__(self, n_harmonic_functions=60, omega0=0.1):
        """
        Given an input tensor `x` of shape [minibatch, ... , dim],
        the harmonic embedding layer converts each feature
        in `x` into a series of harmonic features `embedding`
        as follows:
            embedding[..., i*dim:(i+1)*dim] = [
                sin(x[..., i]),
                sin(2*x[..., i]),
                sin(4*x[..., i]),
                ...
                sin(2**(self.n_harmonic_functions-1) * x[..., i]),
                cos(x[..., i]),
                cos(2*x[..., i]),
                cos(4*x[..., i]),
                ...
                cos(2**(self.n_harmonic_functions-1) * x[..., i])
            ]
            
        Note that `x` is also premultiplied by `omega0` before
        evaluating the harmonic functions.
        """
        super().__init__()
        self.register_buffer(
            'frequencies',
            omega0 * (2.0 ** torch.arange(n_harmonic_functions)),
        )
    def forward(self, x):
        """
        Args:
            x: tensor of shape [..., dim]
        Returns:
            embedding: a harmonic embedding of `x`
                of shape [..., n_harmonic_functions * dim * 2]
        """
        embed = (x[..., None] * self.frequencies).view(*x.shape[:-1], -1)
        return torch.cat((embed.sin(), embed.cos()), dim=-1)


class NeuralRadianceField(torch.nn.Module):
    def __init__(self, n_harmonic_functions=60, n_hidden_neurons=256):
        super().__init__()
        """
        Args:
            n_harmonic_functions: The number of harmonic functions
                used to form the harmonic embedding of each point.
            n_hidden_neurons: The number of hidden units in the
                fully connected layers of the MLPs of the model.
        """
        
        # The harmonic embedding layer converts input 3D coordinates
        # to a representation that is more suitable for
        # processing with a deep neural network.
        self.harmonic_embedding = HarmonicEmbedding(n_harmonic_functions)
        
        # The dimension of the harmonic embedding.
        embedding_dim = n_harmonic_functions * 2 * 3
        
        # self.mlp is a simple 2-layer multi-layer perceptron
        # which converts the input per-point harmonic embeddings
        # to a latent representation.
        # Not that we use Softplus activations instead of ReLU.
        self.mlp = torch.nn.Sequential(
            torch.nn.Linear(embedding_dim, n_hidden_neurons),
            torch.nn.Softplus(beta=10.0),
            torch.nn.Linear(n_hidden_neurons, n_hidden_neurons),
            torch.nn.Softplus(beta=10.0),
        )        
        
        # Given features predicted by self.mlp, self.color_layer
        # is responsible for predicting a 3-D per-point vector
        # that represents the RGB color of the point.
        self.color_layer = torch.nn.Sequential(
            torch.nn.Linear(n_hidden_neurons + embedding_dim, n_hidden_neurons),
            torch.nn.Softplus(beta=10.0),
            torch.nn.Linear(n_hidden_neurons, 3),
            torch.nn.Sigmoid(),
            # To ensure that the colors correctly range between [0-1],
            # the layer is terminated with a sigmoid layer.
        )  
        
        # The density layer converts the features of self.mlp
        # to a 1D density value representing the raw opacity
        # of each point.
        self.density_layer = torch.nn.Sequential(
            torch.nn.Linear(n_hidden_neurons, 1),
            torch.nn.Softplus(beta=10.0),
            # Sofplus activation ensures that the raw opacity
            # is a non-negative number.
        )
        
        # We set the bias of the density layer to -1.5
        # in order to initialize the opacities of the
        # ray points to values close to 0. 
        # This is a crucial detail for ensuring convergence
        # of the model.
        self.density_layer[0].bias.data[0] = -1.5        
                
    def _get_densities(self, features):
        """
        This function takes `features` predicted by `self.mlp`
        and converts them to `raw_densities` with `self.density_layer`.
        `raw_densities` are later mapped to [0-1] range with
        1 - inverse exponential of `raw_densities`.
        """
        raw_densities = self.density_layer(features)
        return 1 - (-raw_densities).exp()
    
    def _get_colors(self, features, rays_directions):
        """
        This function takes per-point `features` predicted by `self.mlp`
        and evaluates the color model in order to attach to each
        point a 3D vector of its RGB color.
        
        In order to represent viewpoint dependent effects,
        before evaluating `self.color_layer`, `NeuralRadianceField`
        concatenates to the `features` a harmonic embedding
        of `ray_directions`, which are per-point directions 
        of point rays expressed as 3D l2-normalized vectors
        in world coordinates.
        """
        spatial_size = features.shape[:-1]
        
        # Normalize the ray_directions to unit l2 norm.
        rays_directions_normed = torch.nn.functional.normalize(
            rays_directions, dim=-1
        )
        
        # Obtain the harmonic embedding of the normalized ray directions.
        rays_embedding = self.harmonic_embedding(
            rays_directions_normed
        )
        
        # Expand the ray directions tensor so that its spatial size
        # is equal to the size of features.
        rays_embedding_expand = rays_embedding[..., None, :].expand(
            *spatial_size, rays_embedding.shape[-1]
        )
        
        # Concatenate ray direction embeddings with 
        # features and evaluate the color model.
        color_layer_input = torch.cat(
            (features, rays_embedding_expand),
            dim=-1
        )
        return self.color_layer(color_layer_input)
    
  
    def forward(
        self, 
        ray_bundle: RayBundle,
        **kwargs,
    ):
        """
        The forward function accepts the parametrizations of
        3D points sampled along projection rays. The forward
        pass is responsible for attaching a 3D vector
        and a 1D scalar representing the point's 
        RGB color and opacity respectively.
        
        Args:
            ray_bundle: A RayBundle object containing the following variables:
                origins: A tensor of shape `(minibatch, ..., 3)` denoting the
                    origins of the sampling rays in world coords.
                directions: A tensor of shape `(minibatch, ..., 3)`
                    containing the direction vectors of sampling rays in world coords.
                lengths: A tensor of shape `(minibatch, ..., num_points_per_ray)`
                    containing the lengths at which the rays are sampled.

        Returns:
            rays_densities: A tensor of shape `(minibatch, ..., num_points_per_ray, 1)`
                denoting the opacity of each ray point.
            rays_colors: A tensor of shape `(minibatch, ..., num_points_per_ray, 3)`
                denoting the color of each ray point.
        """
        # We first convert the ray parametrizations to world
        # coordinates with `ray_bundle_to_ray_points`.
        rays_points_world = ray_bundle_to_ray_points(ray_bundle)
        # rays_points_world.shape = [minibatch x ... x 3]
        # FIXME: offset
        rays_points_world[...,2] -= 2

        # For each 3D world coordinate, we obtain its harmonic embedding.
        embeds = self.harmonic_embedding(
            rays_points_world
        )
        # embeds.shape = [minibatch x ... x self.n_harmonic_functions*6]
        
        # self.mlp maps each harmonic embedding to a latent feature space.
        features = self.mlp(embeds)
        # features.shape = [minibatch x ... x n_hidden_neurons]
        
        # Finally, given the per-point features, 
        # execute the density and color branches.
        
        rays_densities = self._get_densities(features)
        # rays_densities.shape = [minibatch x ... x 1]

        rays_colors = self._get_colors(features, ray_bundle.directions)
        # rays_colors.shape = [minibatch x ... x 3]
        
        return rays_densities, rays_colors
    
    def batched_forward(
        self, 
        ray_bundle: RayBundle,
        n_batches: int = 16,
        **kwargs,        
    ):
        """
        This function is used to allow for memory efficient processing
        of input rays. The input rays are first split to `n_batches`
        chunks and passed through the `self.forward` function one at a time
        in a for loop. Combined with disabling PyTorch gradient caching
        (`torch.no_grad()`), this allows for rendering large batches
        of rays that do not all fit into GPU memory in a single forward pass.
        In our case, batched_forward is used to export a fully-sized render
        of the radiance field for visualization purposes.
        
        Args:
            ray_bundle: A RayBundle object containing the following variables:
                origins: A tensor of shape `(minibatch, ..., 3)` denoting the
                    origins of the sampling rays in world coords.
                directions: A tensor of shape `(minibatch, ..., 3)`
                    containing the direction vectors of sampling rays in world coords.
                lengths: A tensor of shape `(minibatch, ..., num_points_per_ray)`
                    containing the lengths at which the rays are sampled.
            n_batches: Specifies the number of batches the input rays are split into.
                The larger the number of batches, the smaller the memory footprint
                and the lower the processing speed.

        Returns:
            rays_densities: A tensor of shape `(minibatch, ..., num_points_per_ray, 1)`
                denoting the opacity of each ray point.
            rays_colors: A tensor of shape `(minibatch, ..., num_points_per_ray, 3)`
                denoting the color of each ray point.

        """

        # Parse out shapes needed for tensor reshaping in this function.
        n_pts_per_ray = ray_bundle.lengths.shape[-1]  
        spatial_size = [*ray_bundle.origins.shape[:-1], n_pts_per_ray]

        # Split the rays to `n_batches` batches.
        tot_samples = ray_bundle.origins.shape[:-1].numel()
        batches = torch.chunk(torch.arange(tot_samples), n_batches)

        # For each batch, execute the standard forward pass.
        batch_outputs = [
            self.forward(
                RayBundle(
                    origins=ray_bundle.origins.view(-1, 3)[batch_idx],
                    directions=ray_bundle.directions.view(-1, 3)[batch_idx],
                    lengths=ray_bundle.lengths.view(-1, n_pts_per_ray)[batch_idx],
                    xys=None,
                )
            ) for batch_idx in batches
        ]
        
        # Concatenate the per-batch rays_densities and rays_colors
        # and reshape according to the sizes of the inputs.
        rays_densities, rays_colors = [
            torch.cat(
                [batch_output[output_i] for batch_output in batch_outputs], dim=0
            ).view(*spatial_size, -1) for output_i in (0, 1)
        ]
        return rays_densities, rays_colors


def get_nerf(nerf_ckpt):
    neural_radiance_field = NeuralRadianceField().to(device)
    neural_radiance_field.load_state_dict(torch.load(nerf_ckpt))
    return neural_radiance_field

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


def get_nerf_renderer(image_size):
    # Our rendered scene is centered around (0,0,0) 
    # and is enclosed inside a bounding box
    # whose side is roughly equal to 3.0 (world units).
    volume_extent_world = 3.0


    raysampler_grid = NDCMultinomialRaysampler(
        image_height=image_size[0],
        image_width=image_size[1],
        n_pts_per_ray=64,
        min_depth=0.1,
        max_depth=volume_extent_world,
    )

    raymarcher = EmissionAbsorptionRaymarcher()

    # Finally, instantiate the implicit renders
    # for both raysamplers.
    renderer_grid = ImplicitRenderer(
        raysampler=raysampler_grid, raymarcher=raymarcher,
    ).to(device)
    
    return renderer_grid


    
class Pytorch3DImplicitActor(object):
    def __init__(self, single_pass=False):
        self.implict_model = get_nerf('data/implicit/nerf_cow.pt')
        self.renderer_grid = get_nerf_renderer((128,128))

    def init_gl(self,  height, width):
        self.width = width
        self.height = height

        
    def get_imdata(self, cameras, height=None, width=None):
        with torch.no_grad():
            images_raw = self.renderer_grid(
                cameras,
                volumetric_function=self.implict_model.batched_forward
            )[0]
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
    parser = argparse.ArgumentParser(description='Pytorch3D implicit actor argument parser')
    parser.add_argument('--port',
                         type=int,
                         help='port (required by RemoteActorClient)',
                         default=9999)
    actor = Pytorch3DImplicitActor()
    args = parser.parse_args()

    actor.init_gl(1800,1800) # fake image size
    actor.run_as_server(port=args.port)


if __name__ == '__main__':
    main()
   