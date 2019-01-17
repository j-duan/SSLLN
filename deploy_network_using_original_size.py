import os, time, math
import numpy as np, nibabel as nib, pandas as pd
import tensorflow as tf
from image_utils import *


def findLandmarksLocations(pred, x_pre, X, y_pre, Y, z_pre, z_post, z1_, z1, Z, nim):

    # map back to original size 
    if Z < 80:
        pred = pred[x_pre:x_pre + X, y_pre:y_pre + Y, z1_-z1:z1_-z1 + Z]
    else:
        pred = pred[x_pre:x_pre + X, y_pre:y_pre + Y, :]
        pred = np.pad(pred, ((0, 0), (0, 0), (z_pre, z_post)), 'constant')
    
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

                                                                                                                                                                                                                            
if __name__ == '__main__':
       
    model_path  = '/homes/jduan/Desktop/Multitask25D/3datlas_segt_ldmk_dd_1v0/3datlas_segt_ldmk_dd_1v0.ckpt-300'
    test_dir    = '/vol/biomedic2/jduan/3datlas/validation'
    ref_dir     = '/homes/jduan/Desktop/ldmk'     

    with tf.Session() as sess:
         
        sess.run(tf.global_variables_initializer())
        
        # Import the computation graph and restore the variable values
        saver = tf.train.import_meta_graph('{0}.meta'.format(model_path))
        saver.restore(sess, '{0}'.format(model_path))

        print('Start evaluating on the test set ...')
        start_time = time.time()

        # Process each subject subdirectory
        for data in sorted(os.listdir(test_dir)):                                                                                                              
            print(data)
            data_dir = os.path.join(test_dir, data)
            data_dir = os.path.join(data_dir, 'sizes')
            
            if not os.path.isdir(data_dir):
                print('  {0} is not a valid directory, Skip'.format(data_dir))
                continue

            for fr in ['ED']:
                image_name = '{0}/lvsa_{1}_SR_cropped.nii.gz'.format(data_dir, fr)

                # Read the image
                print('  Reading {} ...'.format(image_name))
                nim = nib.load(image_name)
                image = nim.get_data()

                if image.ndim == 4: image = np.squeeze(image, axis=-1).astype(np.int16)             
                if image.ndim == 2: image = np.expand_dims(image, axis=2)
                 
                print('  Segmenting and landmarks detection {} frame ...'.format(fr))
                start_seg_time = time.time()

                # Intensity rescaling
                image = rescale_intensity(image, (1, 99))
                
                # Pad the image size to be a factor of 16 so that the downsample and upsample procedures
                # in the network will result in the same image size at each resolution level.
                X, Y, Z = image.shape
                
                n_slices = 64
                X2, Y2 = int(math.ceil(X / 16.0)) * 16, int(math.ceil(Y / 16.0)) * 16
                x_pre,  y_pre,  z_pre  = int((X2 - X) / 2), int((Y2 - Y) / 2), int((Z - n_slices) / 2)
                x_post, y_post, z_post = (X2 - X) - x_pre, (Y2 - Y) - y_pre, (Z - n_slices) - z_pre
                
                z1,  z2  = int(Z/2) - int(n_slices/2), int(Z/2) + int(n_slices/2)
                z1_, z2_ = max(z1, 0), min(z2, Z)
                image = image[:, :, z1_: z2_]
                image = np.pad(image, ((x_pre, x_post), (y_pre, y_post), (z1_- z1, z2 - z2_)), 'constant')

                # extend batch for network requirement 
                image = np.expand_dims(image, axis=0)
                
                pred_ldmk, pred_segt = sess.run(['pred_ldmk:0', 'pred_segt:0'], feed_dict={'image:0': image, 'training:0': False})
                     
                seg_time = time.time() - start_seg_time
                print('  Segmentation and detection time = {:3f}s'.format(seg_time))
                
                # Transpose and crop the segmentation to recover the original size
                pred_segt = np.squeeze(pred_segt, axis=0).astype(np.int16)
                
                # map back to original size 
                if Z < 64:
                    pred_segt = pred_segt[x_pre:x_pre + X, y_pre:y_pre + Y, z1_-z1:z1_-z1 + Z]
                else:
                    pred_segt = pred_segt[x_pre:x_pre + X, y_pre:y_pre + Y, :]
                    pred_segt = np.pad(pred_segt, ((0, 0), (0, 0), (z_pre, z_post)), 'constant')

                nim2 = nib.Nifti1Image(pred_segt, nim.affine)
                nim2.header['pixdim'] = nim.header['pixdim']
                nib.save(nim2, '{0}/25D.nii.gz'.format(data_dir))

                # Transpose and crop the segmentation to recover the original size
                pred_ldmk = np.squeeze(pred_ldmk, axis=0).astype(np.int16)
            
#                # predicit both rasterized landmarks position and points position
#                pred_ldmk, worldCoord = findLandmarksLocations(pred_ldmk, x_pre, X, y_pre, Y, z_pre, z_post, z1_, z1, Z, nim)
#                                            
#                nim2 = nib.Nifti1Image(pred_ldmk, nim.affine)
#                nim2.header['pixdim'] = nim.header['pixdim']
#                nib.save(nim2, '{0}/landmarks_.nii.gz'.format(data_dir))
#                np.savetxt('{0}/landmarks.txt'.format(data_dir), worldCoord, fmt='%f')
#                os.system('txt2cardiacvtk {0}/landmarks.txt {1}/landmarks.vtk {0}/landmarks.vtk'.format(data_dir, ref_dir))
#                os.system('rm {0}/landmarks.txt'.format(data_dir))
          
                
            
