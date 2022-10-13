#!/bin/env python

from glfw_app import GlfwApp
from renderer import Pytorch3DOpenVrGlRenderer, OpenVrGlRenderer
from actor import ColorCubeActor, Pytorch3DMeshActor, Pytorch3DPointActor, Pytorch3DVolumeActor, Pytorch3DImplicitActor, RemoteActorClient

import openvr
import numpy as np
import argparse

"""
Minimal glfw programming example which colored OpenGL cube scene that can be closed by pressing ESCAPE.
"""


def matrixForOpenVrMatrix(mat):
    if len(mat.m) == 4: # HmdMatrix44_t?
        result = np.matrix( 
                ((mat.m[0][0], mat.m[1][0], mat.m[2][0], mat.m[3][0]),
                 (mat.m[0][1], mat.m[1][1], mat.m[2][1], mat.m[3][1]), 
                 (mat.m[0][2], mat.m[1][2], mat.m[2][2], mat.m[3][2]), 
                 (mat.m[0][3], mat.m[1][3], mat.m[2][3], mat.m[3][3]),)
            , np.float32)
    elif len(mat.m) == 3: # HmdMatrix34_t?
        result = np.matrix(
                ((mat.m[0][0], mat.m[1][0], mat.m[2][0], 0.0),
                 (mat.m[0][1], mat.m[1][1], mat.m[2][1], 0.0), 
                 (mat.m[0][2], mat.m[1][2], mat.m[2][2], 0.0), 
                 (mat.m[0][3], mat.m[1][3], mat.m[2][3], 1.0),)
            , np.float32)
    return result



def main():
    parser = argparse.ArgumentParser(description='COLMAPSLAM argument parser')
    parser.add_argument('--actor',
                         type=str,
                         help='actor type',
                         default=None)
    parser.add_argument('--host_ip',
                         type=str,
                         help='host IP (required by RemoteActorClient)',
                         default=None)
    parser.add_argument('--port',
                         type=int,
                         help='port (required by RemoteActorClient)',
                         default=None)
                         
    args = parser.parse_args()
    single_pass = False

    if args.actor == 'cube':
        actor = ColorCubeActor()
    elif args.actor == 'mesh':
        actor = Pytorch3DMeshActor(single_pass=single_pass)
    elif args.actor == 'point':
        actor = Pytorch3DPointActor(single_pass=single_pass)
    elif args.actor == 'volume':
        actor = Pytorch3DVolumeActor(single_pass=single_pass)
    elif args.actor == 'implicit':
        actor = Pytorch3DImplicitActor(single_pass=single_pass)
    elif args.actor == 'remote':
        assert args.host_ip is not None, 'Need to provide host_ip'
        assert args.port is not None, 'Need to provide port'
        actor = RemoteActorClient(host_ip=args.host_ip, port=args.port)
    else:
        print(f"Error: Actor [{args.actor}] is not implemented")

    # actor = ColorCubeActor()
    # actor = Pytorch3DMeshActor(single_pass=False)
    # actor = Pytorch3DPointActor(single_pass=True)
    # actor = Pytorch3DMeshActor(single_pass=single_pass)
    # actor = Pytorch3DPointActor(single_pass=single_pass)
    # actor = Pytorch3DVolumeActor(single_pass=single_pass)
    # actor = RemoteActorClient(host_ip="192.168.137.1", port=9999)
    if args.actor == 'cube':
        renderer = OpenVrGlRenderer(actor)
    else:
        renderer = Pytorch3DOpenVrGlRenderer(actor, single_pass=single_pass)

    with GlfwApp(renderer, "glfw OpenVR for Pytorch3D") as glfwApp:
        glfwApp.run_loop()


if __name__ == "__main__":
    main()
    