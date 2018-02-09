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

def evaluate(X_data, y_data):
    num_examples = len(X_data)
    total_accuracy = 0
    sess = tf.get_default_session()
    for offset in range(0, num_examples, BATCH_SIZE):
        batch_x, batch_y = X_data[offset:offset+BATCH_SIZE], y_data[offset:offset+BATCH_SIZE]
        accuracy = sess.run(accuracy_operation, feed_dict={x: batch_x, y: batch_y, keep_prob: 1.0, keep_prob_conv: 1.0})
        total_accuracy += (accuracy * len(batch_x))
    return total_accuracy / num_examples


def predict(X_data):
    num_examples = len(X_data)
    sess = tf.get_default_session()
    predicted_proba = list()
    for offset in range(0, num_examples, BATCH_SIZE):
        batch_x = X_data[offset:offset+BATCH_SIZE]
        predicted_proba.extend( sess.run(predict_proba_operation, feed_dict={x: batch_x, keep_prob: 1.0, keep_prob_conv:1}))
    return predicted_proba


def my_test(lr):



    #Definizione dei placeholder
    x = tf.placeholder(tf.float32, (None, 32, 32, 1))
    y = tf.placeholder(tf.int32, (None))
    keep_prob = tf.placeholder(tf.float32)
    keep_prob_conv = tf.placeholder(tf.float32) # usato da una net, forse verrà rimossa

    #Restituisce un tensore (a valori binari) contenente valori tutti posti a 0 tranne uno.
    one_hot_y = tf.one_hot(y, 43)


    logits = LeNet_adv(x, keep_prob)


    #softmax_cross_entropy_with_logits(_sentinel, labels, logits, dim, name)
    cross_entropy = tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=one_hot_y)
    loss_operation = tf.reduce_mean(cross_entropy)
    optimizer = tf.train.AdamOptimizer(learning_rate = lr)
    training_operation = optimizer.minimize(loss_operation)
    predict_operation = tf.argmax(logits, 1)
    predict_proba_operation = tf.nn.softmax(logits=logits)

    correct_prediction = tf.equal(tf.argmax(logits, 1), tf.argmax(one_hot_y, 1))
    accuracy_operation = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

    correct_prediction = tf.equal(tf.argmax(logits, 1), tf.argmax(one_hot_y, 1))
    accuracy_operation = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))





    


    with tf.Session() as sess:
        saver.restore(sess, tf.train.latest_checkpoint(newpath))

        predicted_proba = np.vstack(predict(images))

        print('Accuracy Model On Internet Images: {}'.format(evaluate(images, labels_wild)))


    for true_label,row in zip(labels_wild,predicted_proba):
        top5k = np.argsort(row)[::-1][:5]
        top5p = np.sort(row)[::-1][:5]
        print('Top 5 Labels for image \'{}\':'.format(true_label))
        for k,p in zip(top5k,top5p):
              print(' - \'{}\' with prob = {:.4f} '.format(k, p))
