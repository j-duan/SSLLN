import tensorflow as tf


def FCN(image, segt_class, ldmk_class, n_slice, n_filter, training, same_dim=32, fc=64):
    """
        Build a fully convolutional network for segmenting an input image
        into n_class classes and return the logits map.
    """
    
    net = {}
    tmp = image; # NHWC (Batch_size, Height, Width, Channel=n_slice)
    
    # block 1
    tmp = conv2d_bn_relu(tmp, filters=n_filter[0], training=training)
    net['conv1']  = conv2d_bn_relu(tmp, filters=n_filter[0], training=training)
   
    # block 2
    tmp = conv2d_bn_relu(net['conv1'], filters=n_filter[1], training=training, strides=2)
    net['conv2']  = conv2d_bn_relu(tmp, filters=n_filter[1], training=training)
    
    # block 3
    tmp = conv2d_bn_relu(net['conv2'], filters=n_filter[2], training=training, strides=2)
    tmp = conv2d_bn_relu(tmp, filters=n_filter[2], training=training)
    net['conv3'] = conv2d_bn_relu(tmp, filters=n_filter[2], training=training)  
    
    # block 4
    tmp = conv2d_bn_relu(net['conv3'], filters=n_filter[3], training=training, strides=2)
    tmp = conv2d_bn_relu(tmp, filters=n_filter[3], training=training)
    net['conv4'] = conv2d_bn_relu(tmp, filters=n_filter[3], training=training)
   
    # block 5
    tmp = conv2d_bn_relu(net['conv4'], filters=n_filter[4], training=training, strides=2)
    tmp = conv2d_bn_relu(tmp, filters=n_filter[4], training=training)
    net['conv5'] = conv2d_bn_relu(tmp, filters=n_filter[4], training=training)
       
    # same dimension
    net['conv1_up'] = conv2d_bn_relu(net['conv1'], filters=same_dim, training=training) 
    
    tmp = conv2d_bn_relu(net['conv2'], filters=same_dim, training=training)  
    net['conv2_up'] = tf.layers.conv2d_transpose(tmp, filters=same_dim, kernel_size=3,  strides=2,  padding='same')
    
    tmp = conv2d_bn_relu(net['conv3'], filters=same_dim, training=training) 
    net['conv3_up'] = tf.layers.conv2d_transpose(tmp, filters=same_dim, kernel_size=7,  strides=4,  padding='same')
    
    tmp = conv2d_bn_relu(net['conv4'], filters=same_dim, training=training) 
    net['conv4_up'] = tf.layers.conv2d_transpose(tmp, filters=same_dim, kernel_size=15, strides=8,  padding='same')
    
    tmp = conv2d_bn_relu(net['conv5'], filters=same_dim, training=training)  
    net['conv5_up'] = tf.layers.conv2d_transpose(tmp, filters=same_dim, kernel_size=31, strides=16, padding='same')
   
    # final
    tmp = tf.concat([net['conv1_up'], net['conv2_up'], net['conv3_up'], net['conv4_up'], net['conv5_up']], axis=-1)
    tmp = conv2d_bn_relu(tmp, filters=fc, training=training) 
    tmp = conv2d_bn_relu(tmp, filters=fc, training=training) 
    
    logits_segt = tf.layers.conv2d(tmp, filters=segt_class*n_slice, kernel_size=1, padding='same')
    logits_segt = tf.reshape(logits_segt, [tf.shape(logits_segt)[0], tf.shape(logits_segt)[1], tf.shape(logits_segt)[2], n_slice, segt_class])
    
    logits_ldmk = tf.layers.conv2d(tmp, filters=ldmk_class*n_slice, kernel_size=1, padding='same')
    logits_ldmk = tf.reshape(logits_ldmk, [tf.shape(logits_ldmk)[0], tf.shape(logits_ldmk)[1], tf.shape(logits_ldmk)[2], n_slice, ldmk_class])
    
    return logits_segt, logits_ldmk


def conv2d_bn_relu(x, filters, training, kernel_size=3, strides=1):
    """ Basic Conv + BN + ReLU unit """
   
    x_conv = tf.layers.conv2d(x, filters=filters, kernel_size=kernel_size, strides=strides,
                              padding='same', use_bias=False)
    x_bn = tf.layers.batch_normalization(x_conv, training=training)
    x_relu = tf.nn.relu(x_bn)
    
    return x_relu