import numpy as np 
import nibabel as nib
from scipy import ndimage
from image_utils import *
import SimpleITK as sitk
import math, random
from multiprocessing import Pool
from functools import partial


def resample(image, transform, interpolator):
    reference_image = image
    default_value = 0.0
    return sitk.Resample(image, reference_image, transform, interpolator, default_value)


def data_augmenter(image, segt_label, ldmk_label, shift, rotate, scale, intensity, flip, directions):
    """ perform 3D online affine data augmentation
    """      
    # x is z-dimenson random parameters
    translation = [0, np.clip(random.normalvariate(0, 1), -3, 3) * shift, 
                      np.clip(random.normalvariate(0, 1), -3, 3) * shift]
    rotate_val = np.clip(random.normalvariate(0, 1), -2, 2) * rotate * np.pi/180
    scale_val = 1 + np.clip(random.normalvariate(0, 1), -1.5, 1.5) * scale
    intensity_val = 1 + np.clip(random.normalvariate(0, 1), -3, 3) * intensity
    
    #    translation = [0, np.clip(np.random.normal(), -3, 3) * shift, 
    #                      np.clip(np.random.normal(), -3, 3) * shift]
    #    rotate_val = np.clip(np.random.normal(), -2, 2) * rotate * np.pi/180
    #    scale_val = 1 + np.clip(np.random.normal(), -1.5, 1.5) * scale
    #    intensity_val = 1 + np.clip(np.random.normal(), -3, 3) * intensity
    
    X, Y, Z = image.shape[:3]
    center = (Z/2, X/2, Y/2)
    
    # isotropic scale matrix
    scale = np.array([[scale_val, 0, 0], 
                      [0, scale_val, 0],
                      [0, 0, scale_val]])
    
    # rotation matrices
    rx = np.array([[1, 0, 0], 
                   [0, math.cos(rotate_val), -math.sin(rotate_val)], 
                   [0, math.sin(rotate_val),  math.cos(rotate_val)]])
    
    ry = np.array([[ math.cos(rotate_val), 0, math.sin(rotate_val)], 
                   [0, 1, 0],
                   [-math.sin(rotate_val), 0, math.cos(rotate_val)]])
    
    rz = np.array([[math.cos(rotate_val), -math.sin(rotate_val), 0], 
                   [math.sin(rotate_val),  math.cos(rotate_val), 0], 
                   [0, 0, 1]])
    
    # if no direction, only rotate along z-axis
    if directions == True:
        r_n = np.random.uniform()
        if r_n >= 0.00 and r_n <= 0.333:
            T = np.dot(rx, scale)
        if r_n > 0.333 and r_n <= 0.666:
            T = np.dot(ry, scale)
        if r_n > 0.666:
            T = np.dot(rz, scale)
    else:
        T = np.dot(rx, scale)
        
    # itk requires a vector as input transformation matrix
    T = T.flatten()
    
    # get affine transformation
    affine = sitk.AffineTransform(3)
    affine.SetMatrix(T)
    affine.SetTranslation(translation)
    affine.SetCenter(center)

    # convert to itk format
    image_itk = sitk.GetImageFromArray(image)
    segt_label_itk = sitk.GetImageFromArray(segt_label)
    ldmk_label_itk = sitk.GetImageFromArray(ldmk_label)
    
    # apply affine transformation
    image_itk  = resample(image_itk, affine, sitk.sitkLinear)
    segt_label_itk  = resample(segt_label_itk, affine, sitk.sitkNearestNeighbor)
    ldmk_label_itk  = resample(ldmk_label_itk, affine, sitk.sitkNearestNeighbor)
    
    # convert to image format
    image2  = sitk.GetArrayFromImage(image_itk)
    image2 *= intensity_val
    segt_label2  = sitk.GetArrayFromImage(segt_label_itk)
    ldmk_label2  = sitk.GetArrayFromImage(ldmk_label_itk)
    
    if flip:
        if r_n >= 0.00 and r_n <= 0.333:
            image2 = image2[::-1, :, :]
            segt_label2 = segt_label2[::-1, :, :]
            ldmk_label2 = ldmk_label2[::-1, :, :]
        if r_n > 0.333 and r_n <= 0.666:
            image2 = image2[:, ::-1, :]
            segt_label2 = segt_label2[:, ::-1, :]
            ldmk_label2 = ldmk_label2[:, ::-1, :]
        if r_n > 0.666:
            image2 = image2[:, :, ::-1]
            segt_label2 = segt_label2[:, :, ::-1]     
            ldmk_label2 = ldmk_label2[:, :, ::-1]   
            
    return image2, segt_label2, ldmk_label2


def apply_PC(i, data_list, idx, image_size, n_slice, data_augmentation, shift, rotate, scale, intensity, flip, directions):
    
    image_name, segt_name, ldmk_name = data_list[idx[i]]

    print('  Select {0} {1}'.format(image_name, segt_name, ldmk_name))

    # Read image and label
    image = nib.load(image_name).get_data()
    image = image.astype(np.float32)
    if image.ndim == 4:
        image = np.squeeze(image, axis=-1)
        
    segt_label = nib.load(segt_name).get_data()
    segt_label = segt_label.astype(np.float32)
    if segt_label.ndim == 4:
        segt_label = np.squeeze(segt_label, axis=-1)

    ldmk_label = nib.load(ldmk_name).get_data()
    ldmk_label = ldmk_label.astype(np.float32)
    if ldmk_label.ndim == 4:
        ldmk_label = np.squeeze(ldmk_label, axis=-1)
                       
    # make a cubic for each landmark    
    label_1 = np.roll(ldmk_label,  1, axis=-1)
    label_2 = np.roll(ldmk_label, -1, axis=-1)
    ldmk_label = ldmk_label + label_1 + label_2 
    
    checkAugment = False
    if checkAugment == True:
        image, segt_label, ldmk_label = data_augmenter(image, segt_label, ldmk_label,
                                        shift=shift, rotate=rotate, scale=scale, 
                                        intensity=intensity, flip=flip, directions=directions)
        
        nim = nib.load(image_name)
        nim2 = nib.Nifti1Image(image, nim.affine)        
        nim2.header['pixdim'] = nim.header['pixdim']     
        nib.save(nim2, '/homes/jduan/Desktop/22/image_{}.nii.gz'.format(idx[i]))
        
        nim3 = nib.Nifti1Image(segt_label, nim.affine)        
        nim3.header['pixdim'] = nim.header['pixdim']     
        nib.save(nim3, '/homes/jduan/Desktop/22/segmentation_{}.nii.gz'.format(idx[i]))
        
        nim4 = nib.Nifti1Image(ldmk_label, nim.affine)        
        nim4.header['pixdim'] = nim.header['pixdim']     
        nib.save(nim4, '/homes/jduan/Desktop/22/landmarks_{}.nii.gz'.format(idx[i]))

    # Normalise the image size
    X, Y, Z = image.shape
    cx, cy, cz = int(X / 2), int(Y / 2), int(Z / 2)
    image = crop_3D_image(image, cx, cy, image_size, cz, n_slice)
    segt_label = crop_3D_image(segt_label, cx, cy, image_size, cz, n_slice)
    ldmk_label = crop_3D_image(ldmk_label, cx, cy, image_size, cz, n_slice)
 
    # Perform data augmentation
    if data_augmentation:
        image, segt_label, ldmk_label = data_augmenter(image, segt_label, ldmk_label,
                                        shift=shift, rotate=rotate, scale=scale, 
                                        intensity=intensity, flip=flip, directions=directions)
        
    # Intensity rescaling
    image = rescale_intensity(image, (1.0, 99.0))    
        
    return image, segt_label, ldmk_label


def get_epoch_batch(data_list, batch_size, iteration, idx, image_size=192, 
                    n_slice=60, data_augmentation=False,
                    shift=0.0, rotate=0.0, scale=0.0, 
                    intensity=0.0, flip=False, directions=False):
    
    parallel = True
    if parallel:
    
        print('  data reading and augmentation running on {0} cores'.format(batch_size))

        p = Pool(processes = batch_size) 
        
        # partial only in Python 2.7+
        images, segt_labels, ldmk_labels = zip(*p.map(partial(apply_PC, data_list=data_list, idx=idx, image_size=image_size, 
                                                n_slice=n_slice, data_augmentation=data_augmentation, shift=shift, 
                                                rotate=rotate, scale=scale, intensity=intensity, flip=flip, directions=directions), 
                                                range(iteration*batch_size, (iteration+1)*batch_size)))  
        p.close()
        
    else:
        
        images, segt_labels, ldmk_labels = [], [], []
    
        for i in range(iteration*batch_size, (iteration+1)*batch_size):
            
            image_name, segt_name, ldmk_name = data_list[idx[i]]
    
            print('  Select {0} {1}'.format(image_name, segt_name, ldmk_name))
    
            # Read image and label
            image = nib.load(image_name).get_data()
            image = image.astype(np.float32)
            if image.ndim == 4:
                image = np.squeeze(image, axis=-1)
                
            segt_label = nib.load(segt_name).get_data()
            segt_label = segt_label.astype(np.float32)
            if segt_label.ndim == 4:
                segt_label = np.squeeze(segt_label, axis=-1)
    
            ldmk_label = nib.load(ldmk_name).get_data()
            ldmk_label = ldmk_label.astype(np.float32)
            if ldmk_label.ndim == 4:
                ldmk_label = np.squeeze(ldmk_label, axis=-1)
                               
            tmp_1 = np.roll(ldmk_label,  1, axis=-1)
            tmp_2 = np.roll(ldmk_label, -1, axis=-1)
            ldmk_label = ldmk_label + tmp_1 + tmp_2 
            
            checkAugment = False
            if checkAugment == True:
                image, segt_label, ldmk_label = data_augmenter(image, segt_label, ldmk_label,
                                                shift=shift, rotate=rotate, scale=scale, 
                                                intensity=intensity, flip=flip, directions=directions)
                
                nim = nib.load(image_name)
                nim2 = nib.Nifti1Image(image, nim.affine)        
                nim2.header['pixdim'] = nim.header['pixdim']     
                nib.save(nim2, '/homes/jduan/Desktop/22/image_{}.nii.gz'.format(idx[i]))
                
                nim3 = nib.Nifti1Image(segt_label, nim.affine)        
                nim3.header['pixdim'] = nim.header['pixdim']     
                nib.save(nim3, '/homes/jduan/Desktop/22/segmentation_{}.nii.gz'.format(idx[i]))
                
                nim4 = nib.Nifti1Image(ldmk_label, nim.affine)        
                nim4.header['pixdim'] = nim.header['pixdim']     
                nib.save(nim4, '/homes/jduan/Desktop/22/landmarks_{}.nii.gz'.format(idx[i]))
    
            # Normalise the image size
            X, Y, Z = image.shape
            cx, cy, cz = int(X / 2), int(Y / 2), int(Z / 2)
            image      = crop_3D_image(image, cx, cy, image_size, cz, n_slice)
            segt_label = crop_3D_image(segt_label, cx, cy, image_size, cz, n_slice)
            ldmk_label = crop_3D_image(ldmk_label, cx, cy, image_size, cz, n_slice)
     
            # Perform data augmentation
            if data_augmentation:
                image, segt_label, ldmk_label = data_augmenter(image, segt_label, ldmk_label,
                                                shift=shift, rotate=rotate, scale=scale, 
                                                intensity=intensity, flip=flip, directions=directions)
                
            # Intensity rescaling
            image = rescale_intensity(image, (1.0, 99.0))    
                
            # Append the image slices to the batch
            # Use list for appending, which is much faster than numpy array
            images += [image]
            segt_labels += [segt_label]
            ldmk_labels += [ldmk_label]
    
     # Convert to a numpy array
    images = np.array(images, dtype=np.float32) # batch * height * width * channels (=slices)
    segt_labels = np.array(segt_labels, dtype=np.int32) # batch * height * width * channels (=slices)
    ldmk_labels = np.array(ldmk_labels, dtype=np.int32) # batch * height * width * channels (=slices)

    return images, segt_labels, ldmk_labels
        
        

#def get_epoch_batch(data_list, batch_size, iteration, idx, image_size=192, 
#                    n_slice=60, data_augmentation=False,
#                    shift=0.0, rotate=0.0, scale=0.0, 
#                    intensity=0.0, flip=False, directions=False):
#    
#    images, segt_labels, ldmk_labels = [], [], []
#    
#    for i in range(iteration*batch_size, (iteration+1)*batch_size):
#        
#        image_name, segt_name, ldmk_name = data_list[idx[i]]
#
#        print('  Select {0} {1}'.format(image_name, segt_name, ldmk_name))
#
#        # Read image and label
#        image = nib.load(image_name).get_data()
#        image = image.astype(np.float32)
#        if image.ndim == 4:
#            image = np.squeeze(image, axis=-1)
#            
#        segt_label = nib.load(segt_name).get_data()
#        segt_label = segt_label.astype(np.float32)
#        if segt_label.ndim == 4:
#            segt_label = np.squeeze(segt_label, axis=-1)
#
#        ldmk_label = nib.load(ldmk_name).get_data()
#        ldmk_label = ldmk_label.astype(np.float32)
#        if ldmk_label.ndim == 4:
#            ldmk_label = np.squeeze(ldmk_label, axis=-1)
#                           
#        label_1 = np.roll(ldmk_label,  1, axis=-1)
#        label_2 = np.roll(ldmk_label, -1, axis=-1)
#        ldmk_label = ldmk_label + label_1 + label_2 
#        
#        checkAugment = False
#        if checkAugment == True:
#            image, segt_label, ldmk_label = data_augmenter(image, segt_label, ldmk_label,
#                                            shift=shift, rotate=rotate, scale=scale, 
#                                            intensity=intensity, flip=flip, directions=directions)
#            
#            nim = nib.load(image_name)
#            nim2 = nib.Nifti1Image(image, nim.affine)        
#            nim2.header['pixdim'] = nim.header['pixdim']     
#            nib.save(nim2, '/homes/jduan/Desktop/22/image_{}.nii.gz'.format(idx[i]))
#            
#            nim3 = nib.Nifti1Image(segt_label, nim.affine)        
#            nim3.header['pixdim'] = nim.header['pixdim']     
#            nib.save(nim3, '/homes/jduan/Desktop/22/segmentation_{}.nii.gz'.format(idx[i]))
#            
#            nim4 = nib.Nifti1Image(ldmk_label, nim.affine)        
#            nim4.header['pixdim'] = nim.header['pixdim']     
#            nib.save(nim4, '/homes/jduan/Desktop/22/landmarks_{}.nii.gz'.format(idx[i]))
#
#        # Normalise the image size
#        X, Y, Z = image.shape
#        cx, cy, cz = int(X / 2), int(Y / 2), int(Z / 2)
#        image = crop_3D_image(image, cx, cy, image_size, cz, n_slice)
#        segt_label = crop_3D_image(segt_label, cx, cy, image_size, cz, n_slice)
#        ldmk_label = crop_3D_image(ldmk_label, cx, cy, image_size, cz, n_slice)
# 
#        # Perform data augmentation
#        if data_augmentation:
#            image, segt_label, ldmk_label = data_augmenter(image, segt_label, ldmk_label,
#                                            shift=shift, rotate=rotate, scale=scale, 
#                                            intensity=intensity, flip=flip, directions=directions)
#            
#        # Intensity rescaling
#        image = rescale_intensity(image, (1.0, 99.0))    
#            
#        # Append the image slices to the batch
#        # Use list for appending, which is much faster than numpy array
#        images += [image]
#        segt_labels += [segt_label]
#        ldmk_labels += [ldmk_label]
#
#    # Convert to a numpy array
#    images = np.array(images, dtype=np.float32) # batch * height * width * channels (=slices)
#    segt_labels = np.array(segt_labels, dtype=np.int32) # batch * height * width * channels (=slices)
#    ldmk_labels = np.array(ldmk_labels, dtype=np.int32) # batch * height * width * channels (=slices)
#
#    return images, segt_labels, ldmk_labels


#def get_random_batch(filename_list, batch_size, image_size=192, n_slice=60):
#    # Randomly select batch_size images from filename_list
#    n_file = len(filename_list)
#    n_selected = 0
#    images = []
#    labels = []
#
#    while n_selected < batch_size:
#        rand_index = random.randrange(n_file)
#        image_name, label_name = filename_list[rand_index]
#        if os.path.exists(image_name) and os.path.exists(label_name):
#            print('  Select {0} {1}'.format(image_name, label_name))
#
#            # Read image and label
#            image = nib.load(image_name).get_data()
#            image = image.astype(np.float32)
#            if image.ndim == 4:
#                image = np.squeeze(image, axis=-1)
#                
#            label = sitk.GetArrayFromImage(sitk.ReadImage(label_name))
#            label = np.transpose(label, axes=(2, 1, 0))
#            label = label.astype(np.float32)
#            if label.ndim == 4:
#                label = np.squeeze(label, axis=-1)
#            
#            # Normalise the image size
#            X, Y, Z = image.shape
#            cx, cy, cz = int(X / 2), int(Y / 2), int(Z / 2)
#            image = crop_3D_image(image, cx, cy, image_size, cz, n_slice)
#            label = crop_3D_image(label, cx, cy, image_size, cz, n_slice)
#            
#            # Intensity rescaling
#            image = rescale_intensity(image, (1.0, 99.0))
#
#            # Append the image slices to the batch
#            # Use list for appending, which is much faster than numpy array
#            images += [image]
#            labels += [label]
#
#            # Increase the counter
#            n_selected += 1
#
#    # Convert to a numpy array
#    images = np.array(images, dtype=np.float32)
#    labels = np.array(labels, dtype=np.int32)
#    
#    return images, labels