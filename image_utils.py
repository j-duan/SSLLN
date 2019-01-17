import os
import numpy as np
import tensorflow as tf
from scipy import ndimage


def select_training_data(dataset_dir):
    """ Go through each subset (training, validation) under the data directory
        and list the file names and landmarks of the subjects
    """
    data_list = {}
    for k in ['train', 'validation']:
        
        subset_dir = os.path.join(dataset_dir, k)
        
        data_list[k] = []

        for data in sorted(os.listdir(subset_dir)):
            
            data_dir = os.path.join(subset_dir, data)
            
            data_dir = os.path.join(data_dir, 'sizes')
           
            # using ED frame and the corresponding landmarks     
            for fr in ['ED']:
                image_name = '{0}/lvsa_{1}_SR_cropped.nii.gz'.format(data_dir, fr)
                ldmk_name  = '{0}/landmarks.nii.gz'.format(data_dir, fr)
                segt_name  = '{0}/3D_segmentation_{1}_cropped.nii.gz'.format(data_dir, fr)
                
                if os.path.exists(image_name) and os.path.exists(segt_name) and os.path.exists(ldmk_name):
                    data_list[k] += [[image_name, segt_name, ldmk_name]] 
                        
    return data_list



def crop_3D_image(image, cx, cy, size_xy, cz, size_z):
    """ Crop a 3D image using a bounding box centred at (cx, cy, cz) with specified size """
    X, Y, Z = image.shape[:3]
    rxy = int(size_xy / 2)
    r_z = int(size_z  / 2)
    x1, x2 = cx - rxy, cx + rxy
    y1, y2 = cy - rxy, cy + rxy
    z1, z2 = cz - r_z, cz + r_z
    x1_, x2_ = max(x1, 0), min(x2, X)
    y1_, y2_ = max(y1, 0), min(y2, Y)
    z1_, z2_ = max(z1, 0), min(z2, Z)
    # Crop the image
    crop = image[x1_: x2_, y1_: y2_, z1_: z2_]
    # Pad the image if the specified size is larger than the input image size
    if crop.ndim == 3:
        crop = np.pad(crop, ((x1_ - x1, x2 - x2_), (y1_ - y1, y2 - y2_), (z1_ - z1, z2 - z2_)), 'constant')
    elif crop.ndim == 4:
        crop = np.pad(crop, ((x1_ - x1, x2 - x2_), (y1_ - y1, y2 - y2_) , (z1_ - z1, z2 - z2_), (0, 0)), 'constant')
    else:
        print('Error: unsupported dimension, crop.ndim = {0}.'.format(crop.ndim))
        exit(0)
    return crop


def rescale_intensity(image, thres=(1.0, 99.0)):
    """ Rescale the image intensity to the range of [0, 1] """
    val_l, val_h = np.percentile(image, thres)
    image2 = image
    image2[image < val_l] = val_l
    image2[image > val_h] = val_h
    image2 = (image2.astype(np.float32) - val_l) / (val_h - val_l)
    return image2


def padImg(image, X, Y, Z, image_size, n_slices):
    
    x1  = int(X/2) - int(image_size/2)
    x2  = int(X/2) + int(image_size/2)
    x1_ = max(x1, 0)
    x2_ = min(x2, X)
 
    y1  = int(Y/2) - int(image_size/2)
    y2  = int(Y/2) + int(image_size/2)
    y1_ = max(y1, 0)
    y2_ = min(y2, Y)
    
    z1  = int(Z/2) - int(n_slices/2)
    z2  = int(Z/2) + int(n_slices/2)
    z1_ = max(z1, 0)
    z2_ = min(z2, Z)
        
    image = image[x1_ : x2_, y1_ : y2_, z1_: z2_]
    image = np.pad(image, ((x1_- x1, x2 - x2_), (y1_- y1, y2 - y2_), (z1_- z1, z2 - z2_)), 'constant')   
     
    return image


def mapBack(pred, X, Y, Z, image_size, n_slices):
    
    x1  = int(X/2) - int(image_size/2)
    x1_ = max(x1, 0)
     
    y1  = int(Y/2) - int(image_size/2)
    y1_ = max(y1, 0)
    
    z1  = int(Z/2) - int(n_slices/2)
    z1_ = max(z1, 0)
    
    x_pre  = int((X - image_size) / 2)
    x_post = (X - image_size) - x_pre
    
    y_pre  = int((Y - image_size) / 2)
    y_post = (Y - image_size) - y_pre
    
    z_pre  = int((Z - n_slices) / 2)
    z_post = (Z - n_slices) - z_pre
    
    # map back to original size for x, y and z dimensions
    if X < image_size:
        pred = pred[x1_-x1:x1_-x1 + X, :, :]
    else:
        pred = np.pad(pred, ((x_pre, x_post), (0, 0), (0, 0)), 'constant')
        
    if Y < image_size:
        pred = pred[:, y1_-y1:y1_-y1 + Y, :]
    else:
        pred = np.pad(pred, ((0, 0), (y_pre, y_post), (0, 0)), 'constant')
    
    if Z < n_slices:
        pred = pred[:, :, z1_-z1:z1_-z1 + Z]
    else:
        pred = np.pad(pred, ((0, 0), (0, 0), (z_pre, z_post)), 'constant')
        
    return pred


def findLandmarksLocations(pred, X, Y, Z, image_size, n_slices, nim):

    # map back to original size 
    pred = mapBack(pred, X, Y, Z, image_size, n_slices)
    
    # initalise landmark grid size
    X_, Y_, Z_ = np.meshgrid(range(Y), range(X), range(Z))
       
    # initalise landmarks matrix which is a 3 by 6 matrix
    # extract central masses from prediction
    landmarks = np.zeros([3,6])
    for i in range(6):
        centroidY = np.median(X_[pred==i+1]).astype(np.int16)   
        centroidX = np.median(Y_[pred==i+1]).astype(np.int16)   
        centroidZ = np.median(Z_[pred==i+1]).astype(np.int16)   
        landmarks[:,i] = np.transpose([centroidX, centroidY, centroidZ])      
    landmarks = np.vstack([landmarks, [1,1,1,1,1,1]])
    worldCoord = np.dot(nim.affine,landmarks)
    worldCoord = np.delete(worldCoord, (-1), axis=0)
    worldCoord = np.transpose(worldCoord).flatten()
    
    return pred, worldCoord
