import tensorflow as tf

from tensorflow.contrib.layers import flatten

# LeNet advanced multiple dropout
def LeNet_adv(x, keep_prob):

    # Hyperparameters
    mu = 0
    sigma = 0.1

    # SOLUTION: Layer 1: Convolutional. Input = 32x32x1(or3). Output = 28x28x108.
    conv1_W = tf.Variable(tf.truncated_normal(shape=(5, 5, 1, 108), mean=mu, stddev=sigma))
    conv1_b = tf.Variable(tf.zeros(108))
    conv1 = tf.nn.conv2d(x, conv1_W, strides=[1, 1, 1, 1], padding='VALID') + conv1_b

    # SOLUTION: Activation.
    conv1 = tf.nn.relu(conv1)

    # SOLUTION: Pooling. Input = 28x28x108. Output = 14x14x108.
    conv1 = tf.nn.max_pool(conv1, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='VALID')
    # Dropout
    conv1 = tf.nn.dropout(conv1, keep_prob = 0.8)

    # SOLUTION: Layer 2: Convolutional. Input 14x14x108 Output = 10x10x200.
    conv2_W = tf.Variable(tf.truncated_normal(shape=(5, 5, 108, 200), mean=mu, stddev=sigma))
    conv2_b = tf.Variable(tf.zeros(200))
    conv2 = tf.nn.conv2d(conv1, conv2_W, strides=[1, 1, 1, 1], padding='VALID') + conv2_b

    # SOLUTION: Activation.
    conv2 = tf.nn.relu(conv2)

    # SOLUTION: Pooling. Input = 10x10x200. Output = 5x5x200.
    conv2 = tf.nn.max_pool(conv2, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='VALID')
    #conv2 = tf.nn.dropout(conv2, keep_prob)
    # Dropout
    conv2 = tf.nn.dropout(conv2, keep_prob = 0.7)

    # SOLUTION: Flatten. Input = 5x5x200. Output = 5000.
    fc0 = flatten(conv2)

    # SOLUTION: Layer 3: Fully Connected. Input = 5000. Output = 1000.
    fc1_W = tf.Variable(tf.truncated_normal(shape=(5000, 1000), mean=mu, stddev=sigma))
    fc1_b = tf.Variable(tf.zeros(1000))
    fc1 = tf.matmul(fc0, fc1_W) + fc1_b

    # SOLUTION: Activation.
    fc1 = tf.nn.relu(fc1)

    # SOLUTION: Layer 4: Fully Connected. Input = 1000. Output = 200.
    fc2_W = tf.Variable(tf.truncated_normal(shape=(1000, 200), mean=mu, stddev=sigma))
    fc2_b = tf.Variable(tf.zeros(200))
    fc2 = tf.matmul(fc1, fc2_W) + fc2_b

    # SOLUTION: Activation.
    fc2 = tf.nn.relu(fc2)

    # SOLUTION: Layer 5: Fully Connected. Input = 200. Output = 43.
    fc3_W = tf.Variable(tf.truncated_normal(shape=(200, 43), mean=mu, stddev=sigma))
    fc3_b = tf.Variable(tf.zeros(43))

    # Dropout
    fc2 = tf.nn.dropout(fc2, keep_prob = 0.6)
    logits = tf.matmul(fc2, fc3_W) + fc3_b

    return logits

def LeNet(x, keep_prob):
    """
    Implement classic lenet architecture in tensorflow
    """
    mu = 0
    sigma = 0.1

    # Layer 1: Convolutional. Input = (batch_size)x32x32x1. Output = (batch_size)x28x28x6.
    conv1_W = tf.Variable(tf.truncated_normal(shape=(5, 5, 1, 6), mean = mu, stddev = sigma))
    conv1_b = tf.Variable(tf.zeros(6))
    conv1   = tf.nn.conv2d(x, conv1_W, strides=[1, 1, 1, 1], padding='VALID') + conv1_b

    #  Activation.
    conv1 = tf.nn.relu(conv1)

    # Pooling. Input = (batch_size)x28x28x6. Output = (batch_size)x14x14x6.
    conv1 = tf.nn.max_pool(conv1, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='VALID')

    #  Layer 2: Convolutional. Output = (batch_size)x10x10x16.
    conv2_W = tf.Variable(tf.truncated_normal(shape=(5, 5, 6, 16), mean = mu, stddev = sigma))
    conv2_b = tf.Variable(tf.zeros(16))
    conv2   = tf.nn.conv2d(conv1, conv2_W, strides=[1, 1, 1, 1], padding='VALID') + conv2_b

    #  Activation.
    conv2 = tf.nn.relu(conv2)

    # Pooling. Input = (batch_size)x10x10x16. Output = (batch_size)x5x5x16.
    conv2 = tf.nn.max_pool(conv2, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='VALID')

    # SFlatten. Input = (batch_size)x5x5x16. Output = 400.
    fc0   = flatten(conv2)

    # Layer 3: Fully Connected. Input = 400. Output = 120.
    fc1_W = tf.Variable(tf.truncated_normal(shape=(400, 120), mean = mu, stddev = sigma))
    fc1_b = tf.Variable(tf.zeros(120))
    fc1   = tf.matmul(fc0, fc1_W) + fc1_b

    #  Activation.
    fc1    = tf.nn.relu(fc1)

    dr1 = tf.nn.dropout(fc1, keep_prob)

    # Layer 4: Fully Connected. Input = 120. Output = 84.
    fc2_W  = tf.Variable(tf.truncated_normal(shape=(120, 84), mean = mu, stddev = sigma))
    fc2_b  = tf.Variable(tf.zeros(84))
    fc2    = tf.matmul(dr1, fc2_W) + fc2_b

    # Activation.
    fc2    = tf.nn.relu(fc2)


    dr2 = tf.nn.dropout(fc2, keep_prob)

    #  Layer 5: Fully Connected. Input = 84. Output = 43.
    fc3_W  = tf.Variable(tf.truncated_normal(shape=(84, 43), mean = mu, stddev = sigma))
    fc3_b  = tf.Variable(tf.zeros(43))
    logits = tf.matmul(dr2, fc3_W) + fc3_b

    return logits


# # LeNet advanced
# def LeNet_adv(x, keep_prob):
#
#     # Hyperparameters
#     mu = 0
#     sigma = 0.1
#
#     # SOLUTION: Layer 1: Convolutional. Input = 32x32x1(or3). Output = 28x28x108.
#     conv1_W = tf.Variable(tf.truncated_normal(shape=(5, 5, 1, 108), mean=mu, stddev=sigma))
#     conv1_b = tf.Variable(tf.zeros(108))
#     conv1 = tf.nn.conv2d(x, conv1_W, strides=[1, 1, 1, 1], padding='VALID') + conv1_b
#
#     # SOLUTION: Activation.
#     conv1 = tf.nn.relu(conv1)
#
#     # SOLUTION: Pooling. Input = 28x28x108. Output = 14x14x108.
#     conv1 = tf.nn.max_pool(conv1, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='VALID')
#
#     # SOLUTION: Layer 2: Convolutional. Input 14x14x108 Output = 10x10x200.
#     conv2_W = tf.Variable(tf.truncated_normal(shape=(5, 5, 108, 200), mean=mu, stddev=sigma))
#     conv2_b = tf.Variable(tf.zeros(200))
#     conv2 = tf.nn.conv2d(conv1, conv2_W, strides=[1, 1, 1, 1], padding='VALID') + conv2_b
#
#     # SOLUTION: Activation.
#     conv2 = tf.nn.relu(conv2)
#
#     # SOLUTION: Pooling. Input = 10x10x200. Output = 5x5x200.
#     conv2 = tf.nn.max_pool(conv2, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='VALID')
#     conv2 = tf.nn.dropout(conv2, keep_prob)
#
#     # SOLUTION: Flatten. Input = 5x5x200. Output = 5000.
#     fc0 = flatten(conv2)
#
#     # SOLUTION: Layer 3: Fully Connected. Input = 5000. Output = 1000.
#     fc1_W = tf.Variable(tf.truncated_normal(shape=(5000, 1000), mean=mu, stddev=sigma))
#     fc1_b = tf.Variable(tf.zeros(1000))
#     fc1 = tf.matmul(fc0, fc1_W) + fc1_b
#
#     # SOLUTION: Activation.
#     fc1 = tf.nn.relu(fc1)
#
#     # SOLUTION: Layer 4: Fully Connected. Input = 1000. Output = 200.
#     fc2_W = tf.Variable(tf.truncated_normal(shape=(1000, 200), mean=mu, stddev=sigma))
#     fc2_b = tf.Variable(tf.zeros(200))
#     fc2 = tf.matmul(fc1, fc2_W) + fc2_b
#
#     # SOLUTION: Activation.
#     fc2 = tf.nn.relu(fc2)
#
#     # SOLUTION: Layer 5: Fully Connected. Input = 200. Output = 43.
#     fc3_W = tf.Variable(tf.truncated_normal(shape=(200, 43), mean=mu, stddev=sigma))
#     fc3_b = tf.Variable(tf.zeros(43))
#     logits = tf.matmul(fc2, fc3_W) + fc3_b
#
#     return logits


# VGGnet
def VGGnet(x, keep_prob, keep_prob_conv):
    mu = 0
    sigma = 0.1

    # Layer 1: Convolutional. Input = 32x32x1. Output = 32x32x32.
    conv1_W = tf.Variable(tf.truncated_normal(shape=(3, 3, 1, 32), mean = mu, stddev = sigma))
    conv1_b = tf.Variable(tf.zeros(32))
    conv1   = tf.nn.conv2d(x, conv1_W, strides=[1, 1, 1, 1], padding='SAME') + conv1_b

    # ReLu Activation.
    conv1 = tf.nn.relu(conv1)

    # Layer 2: Convolutional. Input = 32x32x32. Output = 32x32x32.
    conv2_W = tf.Variable(tf.truncated_normal(shape=(3, 3, 32, 32), mean = mu, stddev = sigma))
    conv2_b = tf.Variable(tf.zeros(32))
    conv2   = tf.nn.conv2d(conv1, conv2_W, strides=[1, 1, 1, 1], padding='SAME') + conv2_b

    # ReLu Activation.
    conv2 = tf.nn.relu(conv2)

    # Pooling. Input = 32x32x32. Output = 16x16x32.
    conv2 = tf.nn.max_pool(conv2, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='VALID')
    conv2 = tf.nn.dropout(conv2, keep_prob_conv) # dropout

    # Layer 3: Convolutional. Input = 16x16x32. Output = 16x16x64.
    conv3_W = tf.Variable(tf.truncated_normal(shape=(3, 3, 32, 64), mean = mu, stddev = sigma))
    conv3_b = tf.Variable(tf.zeros(64))
    conv3   = tf.nn.conv2d(conv2, conv3_W, strides=[1, 1, 1, 1], padding='SAME') + conv3_b

    # ReLu Activation.
    conv3 = tf.nn.relu(conv3)

    # Layer 4: Convolutional. Input = 16x16x32. Output = 16x16x64.
    conv4_W = tf.Variable(tf.truncated_normal(shape=(3, 3, 64, 64), mean = mu, stddev = sigma))
    conv4_b = tf.Variable(tf.zeros(64))
    conv4   = tf.nn.conv2d(conv3, conv4_W, strides=[1, 1, 1, 1], padding='SAME') + conv4_b

    # ReLu Activation.
    conv4 = tf.nn.relu(conv4)

    # Pooling. Input = 16x16x64. Output = 8x8x64.
    conv4 = tf.nn.max_pool(conv4, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='VALID')
    conv4 = tf.nn.dropout(conv4, keep_prob_conv) # dropout

    # Layer 5: Convolutional. Input = 8x8x64. Output = 8x8x128.
    conv5_W = tf.Variable(tf.truncated_normal(shape=(3, 3, 64, 128), mean = mu, stddev = sigma))
    conv5_b = tf.Variable(tf.zeros(128))
    conv5   = tf.nn.conv2d(conv4, conv5_W, strides=[1, 1, 1, 1], padding='SAME') + conv5_b

    # ReLu Activation.
    conv5 = tf.nn.relu(conv5)

    # Layer 6: Convolutional. Input = 8x8x128. Output = 8x8x128.
    conv6_W = tf.Variable(tf.truncated_normal(shape=(3, 3, 128, 128), mean = mu, stddev = sigma))
    conv6_b = tf.Variable(tf.zeros(128))
    conv6   = tf.nn.conv2d(conv5, conv6_W, strides=[1, 1, 1, 1], padding='SAME') + conv6_b

    # ReLu Activation.
    conv6 = tf.nn.relu(conv6)

    # Pooling. Input = 8x8x128. Output = 4x4x128.
    conv6 = tf.nn.max_pool(conv6, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='VALID')
    conv6 = tf.nn.dropout(conv6, keep_prob_conv) # dropout

    # Flatten. Input = 4x4x128. Output = 2048.
    fc0   = flatten(conv6)

    # Layer 7: Fully Connected. Input = 2048. Output = 128.
    fc1_W = tf.Variable(tf.truncated_normal(shape=(2048, 128), mean = mu, stddev = sigma))
    fc1_b = tf.Variable(tf.zeros(128))
    fc1   = tf.matmul(fc0, fc1_W) + fc1_b

    # ReLu Activation.
    fc1    = tf.nn.relu(fc1)
    fc1    = tf.nn.dropout(fc1, keep_prob) # dropout

    # Layer 8: Fully Connected. Input = 128. Output = 128.
    fc2_W  = tf.Variable(tf.truncated_normal(shape=(128, 128), mean = mu, stddev = sigma))
    fc2_b  = tf.Variable(tf.zeros(128))
    fc2    = tf.matmul(fc1, fc2_W) + fc2_b

    # ReLu Activation.
    fc2    = tf.nn.relu(fc2)
    fc2    = tf.nn.dropout(fc2, keep_prob) # dropout

    # Layer 9: Fully Connected. Input = 128. Output = 43.
    fc3_W  = tf.Variable(tf.truncated_normal(shape=(128, 43), mean = mu, stddev = sigma))
    fc3_b  = tf.Variable(tf.zeros(43))
    logits = tf.matmul(fc2, fc3_W) + fc3_b

    return logits
