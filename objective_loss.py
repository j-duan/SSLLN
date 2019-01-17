import tensorflow as tf

def tf_categorical_accuracy(pred, truth):
    """ Accuracy metric """
    return tf.reduce_mean(tf.cast(tf.equal(pred, truth), dtype=tf.float32))


def tf_categorical_dice(pred, truth, k):
    """ Dice overlap metric for label k """
    A = tf.cast(tf.equal(pred, k), dtype=tf.float32)
    B = tf.cast(tf.equal(truth, k), dtype=tf.float32)
    return 2 * tf.reduce_sum(tf.multiply(A, B)) / (tf.reduce_sum(A) + tf.reduce_sum(B))


def tf_weighted_loss_batch(prob, label_1hot):
    """ weighted cross entropy loss function using batch"""
    prob = tf.clip_by_value(prob, 10e-8, 1.-10e-8)
    count_neg = tf.reduce_sum(1 - label_1hot, [0, 1, 2, 3])
    count_pos = tf.reduce_sum(    label_1hot, [0, 1, 2, 3])
    weights = count_neg / (count_neg + count_pos)  
    label_loss = -label_1hot * tf.log(prob)
    loss = tf.reduce_mean(label_loss * weights)   
    return loss


def tf_mannual_cross_entropy_loss(prob, label_1hot):
    """my own cross entropy loss"""
    prob = tf.clip_by_value(prob, 10e-8, 1.-10e-8)
    loss = -tf.reduce_mean(label_1hot * tf.log(prob))
    return loss


def tf_built_in_sparse_cross_entropy_loss(logits, label_pl):
    """built-in sparse cross entropy loss, requiring non-1-hot label input"""
    label_loss_sparse = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=label_pl, logits=logits)
    loss = tf.reduce_mean(label_loss_sparse) 
    return loss

    
def tf_dice_loss(prob, label_1hot):
    """ Dice loss function"""
    tmp_0 = tf.reduce_sum(prob * label_1hot, [1,2,3])
    tmp_1 = tf.reduce_sum(prob + label_1hot, [1,2,3]) 
    tmp_3 = (2 * tmp_0 ) / (tmp_1 + 1e-7)
    loss  = - tf.reduce_mean(tmp_3)
    return loss


def tf_dice_loss_square(prob, label_1hot):
    """ Dice loss function"""
    tmp_0 = tf.reduce_sum(prob * label_1hot, [1,2,3])
    tmp_1 = tf.reduce_sum(prob * prob + label_1hot * label_1hot, [1,2,3]) 
    tmp_3 = (2 * tmp_0 ) / (tmp_1 + 1e-8)
    loss  = 1 - tf.reduce_mean(tmp_3)
    return loss


def tf_weighted_loss_signle(prob, label_1hot, n_class):
    """ weighted cross entropy loss function using each training label"""
    prob = tf.clip_by_value(prob, 10e-8, 1.-10e-8) 
    y = label_1hot #[B, W, L, H, C] -> {0,1}
    B, W, L, H = tf.shape(y)[0], tf.shape(y)[1], tf.shape(y)[2], tf.shape(y)[3]
    ones = tf.ones([W, L, H, B], tf.float32) #[W, L, H, B] -> {1}
    weights = []
    for i in range(n_class):
        wch_label = y[:, :, :, :, i] #[B, W, L, H] -> {0,1}
        neg_count = tf.reduce_sum(1 - wch_label, [1, 2, 3]) #[B,]
        pos_count = tf.reduce_sum(    wch_label, [1, 2, 3]) #[B,]
        if i == 0:
            wch_weight = 50 * neg_count / (neg_count + pos_count) #[B,] -> [0,1] 
        else: 
            wch_weight = neg_count / (neg_count + pos_count) #[B,] -> [0,1] 
        #[W, L, H, B]*[B,] = [W, L, H, B]^T(3, 0, 1, 2) = [B, W, L, H] -> [0,1] 
        weights += [tf.transpose(ones*wch_weight, [3, 0, 1, 2])] 
    #[[B, W, L, H], [B, W, L, H], [B, W, L, H], [B, W, L, H], ...]
    weights = tf.stack(weights, axis=-1) #[B, W, L, H, C]
    label_loss = weights * label_1hot * tf.log(prob) #[B, W, L, H, C]
    loss = tf.reduce_mean(-label_loss)  #[1,]
    return loss


def tf_built_in_cross_entropy_loss(logits, label_1hot):
    """built-in cross entropy loss, requiring 1-hot label input"""
    label_loss = tf.nn.softmax_cross_entropy_with_logits(labels=label_1hot, logits=logits)
    loss = tf.reduce_mean(label_loss) 
    return loss


def tf_loss_accuary(logits_segt, logits_ldmk, segt_pl, ldmk_pl, segt_class, ldmk_class):
    
    # The softmax probability and the predicted segmentation   
    prob_segt = tf.nn.softmax(logits_segt, name='prob_segt') #([batch, Width, Height, Slice, segt_class])
    pred_segt = tf.cast(tf.argmax(prob_segt, axis=-1), dtype=tf.int32, name='pred_segt') #([batch, Width, Height, Slice])
    
    prob_ldmk = tf.nn.softmax(logits_ldmk, name='prob_ldmk') #([batch, Width, Height, Slice, ldmk_class])
    pred_ldmk = tf.cast(tf.argmax(prob_ldmk, axis=-1), dtype=tf.int32, name='pred_ldmk') #([batch, Width, Height, Slice])
    
    print('prob_segt.name = ' + prob_segt.name)
    print('pred_segt.name = ' + pred_segt.name)
    print('prob_ldmk.name = ' + prob_ldmk.name)
    print('pred_ldmk.name = ' + pred_ldmk.name)
    
    label_1hot = tf.one_hot(indices=segt_pl, depth=segt_class)
    loss_segt = tf_dice_loss_square(prob_segt, label_1hot)
    
    label_1hot = tf.one_hot(indices=ldmk_pl, depth=ldmk_class)
    #    loss_ldmk = tf_dice_loss_square(prob_ldmk, label_1hot)
    loss_ldmk = tf_weighted_loss_signle(prob_ldmk, label_1hot, ldmk_class)


    alfa = 1
    loss = alfa*loss_segt + (1-alfa)*loss_ldmk
    
    # Evaluation metrics
    accuracy_segt = tf_categorical_accuracy(pred_segt, segt_pl)
    accuracy_ldmk = tf_categorical_accuracy(pred_ldmk, ldmk_pl)
    
    #    dice_lv  = tf_categorical_dice(pred, label_pl, 1)
    #    dice_lvmyo = tf_categorical_dice(pred, label_pl, 2)
    #    dice_rv  = tf_categorical_dice(pred, label_pl, 4)
    #    dice_rvmyo  = tf_categorical_dice(pred, label_pl, 3)
    
    return loss, accuracy_segt, accuracy_ldmk
    
    
