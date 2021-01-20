import sys
import numpy as np
import scipy.io as sio
import cv2


import mat73






def main():
    print("Load data..")
    data_dict = mat73.loadmat('../gt/depths.mat')
    depth_image = data_dict['depths']
    d1 = depth_image[:,:,0]
    d1_n = d1 / d1.max()
    # depth_image.shape,  np.min(d1), np.max(d1),d1_n.shape
    
    cv2.imshow("depth raw", d1)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    
    cv2.imshow("depth normal", d1_n)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


    # need to get an image with 3 channels
    d1_n_3 = np.stack((d1_n, d1_n, d1_n), axis=2)
    print("Depth image with 3 channels: shape", d1_n_3.shape)

    d_im = d1_n_3.astype("float64")

def normls_loop(d_im):
    normals = np.array(d_im, dtype="float64")
    h,w,d = d_im.shape
    for i in range(1,w-1):
        for j in range(1,h-1):
            t = np.array([i,j-1,d_im[j-1,i,0]],dtype="float64")
            f = np.array([i-1,j,d_im[j,i-1,0]],dtype="float64")
            c = np.array([i,j,d_im[j,i,0]] , dtype = "float64")
            d = np.cross(f-c,t-c)
            n = d / np.sqrt((np.sum(d**2)))
            normals[j,i,:] = n

    return normals



if __name__ =="__main__":
    main()
