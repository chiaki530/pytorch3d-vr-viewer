from numpysocket import NumpySocket
import numpy as np
from OpenGL.GL import * 
from time import sleep
import cv2

class RemoteActorClient(object):
    def __init__(self, host_ip, port, single_pass=False):
        self.single_pass = single_pass
        self.host_ip = host_ip
        self.port = port

    def init_gl(self, height, width):
        self.width = width
        self.height = height

        # try to connect to server
        self.npSocket = NumpySocket()
        while(True):
            try:
                self.npSocket.connect((self.host_ip, self.port))
                break
            except Exception as e:
                print("connection failed, make sure `server` is running.", self.host_ip, self.port, e)
                sleep(1)
                continue

    def display_gl(self, hmd_pose, eye2hmd, projection, texture_id):
        # encode pose into array and sent
        cam_data = np.zeros((20,4,4)) # BUG: if send to small array, the server side will be block
        cam_data[0,:3,:4] = hmd_pose
        cam_data[1,:3,:4] = eye2hmd
        cam_data[2,:4,:4] = projection
        cam_data[3,0,0] = self.height
        cam_data[3,0,1] = self.width

        self.npSocket.sendall(cam_data)    

        # receive rendered_image_np from server
        imdata_base = np.ones((self.height, self.width,4)).astype(np.uint8)
        imdata = self.npSocket.recv(bufsize=12288)
        # print("client receieve frame", imdata.shape)
        imdata_base[...,:3] = cv2.resize(imdata.astype(np.uint8), (self.width, self.height))
        
        glBindTexture(GL_TEXTURE_2D, texture_id)
        glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA8, self.width, self.height, 0, GL_RGBA, GL_UNSIGNED_BYTE, imdata_base)
    
    def get_stereo_views_np(self, hmd_pose, eye_left, projection_left, eye_right, projection_right):
        pass
            
    def dispose_gl(self):
        pass