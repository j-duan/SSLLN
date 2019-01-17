import os, time, random
import numpy as np, nibabel as nib
import tensorflow as tf
from network_architecture import *
from image_utils import *
from batch import *
from objective_loss import *

""" Training parameters """
FLAGS = tf.app.flags.FLAGS
tf.app.flags.DEFINE_integer('image_size', 112, 'Image size after cropping.')
tf.app.flags.DEFINE_integer('n_slice', 64, 'Image size after cropping.')
tf.app.flags.DEFINE_integer('train_batch_size', 4, 'Number of images for each training batch.')
tf.app.flags.DEFINE_integer('validation_batch_size', 8, 'Number of images for each validation batch.')
tf.app.flags.DEFINE_integer('train_epoch', 300, 'Number of training iterations.')
tf.app.flags.DEFINE_integer('segt_class', 5, 'Number of markers.')
tf.app.flags.DEFINE_integer('ldmk_class', 7, 'Number of markers.')
tf.app.flags.DEFINE_float('learning_rate', 1e-3, 'Learning rate.')
tf.app.flags.DEFINE_string('dataset_dir', '/vol/biomedic2/jduan/3datlas',
                           'Path to the dataset directory, which is split into training, validation '
                           'and test subdirectories.')
tf.app.flags.DEFINE_string('log_dir', '/homes/jduan/Desktop/Multitask25D/saver/log',
                           'Directory for saving the log file.')
tf.app.flags.DEFINE_string('checkpoint_dir', '/homes/jduan/Desktop/Multitask25D/saver/model',
                           'Directory for saving the trained model.')

def main(_):
    """ Main function """

    data_list = select_training_data(FLAGS.dataset_dir)
    
    image_pl = tf.placeholder(tf.float32, shape=[None, None, None, FLAGS.n_slice], name='image')
    ldmk_pl = tf.placeholder(tf.int32, shape=[None, None, None, FLAGS.n_slice], name='label_ldmk')
    segt_pl = tf.placeholder(tf.int32, shape=[None, None, None, FLAGS.n_slice], name='label_segt')
    training_pl = tf.placeholder(tf.bool, shape=[], name='training')
   
    # Print out the placeholders' names, which will be useful when deploying the network
    print('Placeholder image_pl.name = ' + image_pl.name)
    print('Placeholder ldmk_pl.name = ' + ldmk_pl.name)
    print('Placeholder segt_pl.name = ' + segt_pl.name)
    print('Placeholder training_pl.name = ' + training_pl.name)

    # The number of filters at each resolution level
    # Follow the VGG philosophy, increasing the dimension by a factor of 2 for each level
    logits_segt, logits_ldmk = FCN(image_pl, FLAGS.segt_class, FLAGS.ldmk_class, 
                                   n_slice=FLAGS.n_slice, n_filter=[16,32,64,128,256], 
                                   training=training_pl, same_dim=32, fc=64)

    loss, accuracy_segt, accuracy_ldmk = tf_loss_accuary(logits_segt, logits_ldmk, segt_pl, ldmk_pl, 
                                                         FLAGS.segt_class, FLAGS.ldmk_class)
    
    # We need to add the operators associated with batch_normalization to the optimiser, according to
    # https://www.tensorflow.org/api_docs/python/tf/layers/batch_normalization
    update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
    with tf.control_dependencies(update_ops):
        train_op = tf.train.AdamOptimizer(learning_rate=FLAGS.learning_rate).minimize(loss)
    
    # Model name and directory
    model_name = '3datlas_segt_ldmk_dd_1v0'
    model_dir = os.path.join(FLAGS.checkpoint_dir, model_name)
    if not os.path.exists(model_dir):
        os.makedirs(model_dir)

    # Create a logger
    if not os.path.exists(FLAGS.log_dir):
        os.makedirs(FLAGS.log_dir)
    csv_name = os.path.join(FLAGS.log_dir, '{0}_log.csv'.format(model_name))
    f_log = open(csv_name, 'w')
    f_log.write('epoch,time,train_loss,train_acc,test_loss,test_acc,test_dice_lv,test_dice_lv_myo,test_dice_rv,test_dice_rv_myo\n')

    # Start the tensorflow session
    with tf.Session() as sess:
        print('Start training...')
        start_time = time.time()

        # Create a saver
        saver = tf.train.Saver()

        # Initialise variables
        sess.run(tf.global_variables_initializer())
       
        for epoch in range(1, 1 + FLAGS.train_epoch):
            print('epoch {} out of {}: training...'.format(epoch, FLAGS.train_epoch))  
            
            data_len = len(data_list['train'])
            n_per_epoch = int(data_len/FLAGS.train_batch_size)
            idx = random.sample(range(data_len), data_len)
                        
            for iteration in range(n_per_epoch):
                print('  iteration {0} in {1} epoch out of {2}'.format(iteration, epoch, FLAGS.train_epoch))       
                start_time_iter = time.time()
                
                images, segt_labels, ldmk_labels = get_epoch_batch(data_list['train'], FLAGS.train_batch_size, iteration, idx,
                                                   image_size=FLAGS.image_size, n_slice=FLAGS.n_slice, 
                                                   data_augmentation=True, shift=0, rotate=10, scale=0.1, 
                                                   intensity=0.1, flip=False, directions=False)
    
                # Stochastic optimisation using this batch
                _, train_loss, train_acc, train_acc_ = sess.run([train_op, loss, accuracy_segt, accuracy_ldmk], 
                                                                {image_pl: images, segt_pl: segt_labels, 
                                                                 ldmk_pl: ldmk_labels, training_pl: True})
             
                # Print the results for this iteration
                print('  Iteration {} in {} epoch out of {} took {:.3f}s'.format(iteration, epoch, FLAGS.train_epoch, time.time() - start_time_iter))
                print('  training loss:\t\t{:.6f}'.format(train_loss))
                print('  training segmentation accuracy:\t\t{:.2f}%'.format(train_acc * 100))   
                print('  training landmark accuracy:\t\t{:.2f}%'.format(train_acc_ * 100))   
               
            # Print the validation results after one epoach
#            print('epoch {} out of {} is finshed, starting validation...'.format(epoch, FLAGS.train_epoch))
#            images, labels = get_random_batch(data_list['validation'], FLAGS.validation_batch_size, image_size=FLAGS.image_size, n_slice=FLAGS.n_slice)
#    
#            # evalidate the random batch by setting training_pl false
#            validation_loss, validation_acc, validation_dice_lv, validation_dice_lvmyo, validation_dice_rv, validation_dice_rvmyo = \
#            sess.run([loss, accuracy, dice_lv, dice_lvmyo, dice_rv, dice_rvmyo], {image_pl: images, label_pl: labels, training_pl: False})
#
#            # Print the validation results after one epoach
#            print('  finish {} epoch out of {}, taking {:.3f}s'.format(epoch, FLAGS.train_epoch, time.time() - start_time_iter))
#            print('  training loss:\t\t{:.6f}'.format(train_loss))
#            print('  training accuracy:\t\t{:.2f}%'.format(train_acc * 100))   
#            print('  validation loss: \t\t{:.6f}'.format(validation_loss))
#            print('  validation accuracy:\t\t{:.2f}%'.format(validation_acc * 100))
#            print('  validation Dice LV:\t\t{:.6f}'.format(validation_dice_lv))
#            print('  validation Dice LVMyo:\t\t{:.6f}'.format(validation_dice_lvmyo))
#            print('  validation Dice RV:\t\t{:.6f}'.format(validation_dice_rv))
#            print('  validation Dice RVMyo:\t\t{:.6f}\n'.format(validation_dice_rvmyo))
#        
#            f_log.write('{0}, {1}, {2}, {3}, {4}, {5}, {6}, {7}, {8}, {9}\n'.format(
#                     epoch, time.time() - start_time, train_loss, train_acc, validation_loss,
#                     validation_acc, validation_dice_lv, validation_dice_lvmyo, validation_dice_rv, validation_dice_rvmyo))
#            f_log.flush()
                
            # Save models after every 1000 iterations (1 epoch)
            if epoch % 50 == 0:
                saver.save(sess, save_path=os.path.join(model_dir, '{0}.ckpt'.format(model_name)), global_step=epoch)

        # Close the logger and summary writers
        f_log.close()
        print('Training took {:.3f}s in total.\n'.format(time.time() - start_time))


if __name__ == '__main__':
    tf.app.run()
