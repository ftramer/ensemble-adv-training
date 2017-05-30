import keras.backend as K
import numpy as np
import tensorflow as tf
from tensorflow.python.platform import flags
from attack_utils import gen_adv_loss

import time
import sys

FLAGS = flags.FLAGS
EVAL_FREQUENCY = 100


def batch_eval(tf_inputs, tf_outputs, numpy_inputs):
    """
    A helper function that computes a tensor on numpy inputs by batches.
    From: https://github.com/openai/cleverhans/blob/master/cleverhans/utils_tf.py
    """

    n = len(numpy_inputs)
    assert n > 0
    assert n == len(tf_inputs)
    m = numpy_inputs[0].shape[0]
    for i in range(1, n):
        assert numpy_inputs[i].shape[0] == m

    out = []
    for _ in tf_outputs:
        out.append([])

    for start in range(0, m, FLAGS.BATCH_SIZE):
        batch = start // FLAGS.BATCH_SIZE

        # Compute batch start and end indices
        start = batch * FLAGS.BATCH_SIZE
        end = start + FLAGS.BATCH_SIZE
        numpy_input_batches = [numpy_input[start:end]
                               for numpy_input in numpy_inputs]
        cur_batch_size = numpy_input_batches[0].shape[0]
        assert cur_batch_size <= FLAGS.BATCH_SIZE
        for e in numpy_input_batches:
            assert e.shape[0] == cur_batch_size

        feed_dict = dict(zip(tf_inputs, numpy_input_batches))
        feed_dict[K.learning_phase()] = 0
        numpy_output_batches = K.get_session().run(tf_outputs,
                                                   feed_dict=feed_dict)
        for e in numpy_output_batches:
            assert e.shape[0] == cur_batch_size, e.shape
        for out_elem, numpy_output_batch in zip(out, numpy_output_batches):
            out_elem.append(numpy_output_batch)

    out = [np.concatenate(x, axis=0) for x in out]
    for e in out:
        assert e.shape[0] == m, e.shape
    return out


def tf_train(x, y, model, X_train, Y_train, generator, x_advs=None):
    old_vars = set(tf.all_variables())
    train_size = Y_train.shape[0]

    # Generate cross-entropy loss for training
    logits = model(x)
    preds = K.softmax(logits)
    l1 = gen_adv_loss(logits, y, mean=True)

    # add adversarial training loss
    if x_advs is not None:
        idx = tf.placeholder(dtype=np.int32)
        logits_adv = model(tf.stack(x_advs)[idx])
        l2 = gen_adv_loss(logits_adv, y, mean=True)
        loss = 0.5*(l1+l2)
    else:
        l2 = tf.constant(0)
        loss = l1

    optimizer = tf.train.AdamOptimizer().minimize(loss)

    # Run all the initializers to prepare the trainable parameters.
    K.get_session().run(tf.initialize_variables(
        set(tf.all_variables()) - old_vars))
    start_time = time.time()
    print('Initialized!')

    # Loop through training steps.
    num_steps = int(FLAGS.NUM_EPOCHS * train_size + FLAGS.BATCH_SIZE - 1) // FLAGS.BATCH_SIZE

    step = 0
    for (batch_data, batch_labels) \
            in generator.flow(X_train, Y_train, batch_size=FLAGS.BATCH_SIZE):

        if len(batch_data) < FLAGS.BATCH_SIZE:
            k = FLAGS.BATCH_SIZE - len(batch_data)
            batch_data = np.concatenate([batch_data, X_train[0:k]])
            batch_labels = np.concatenate([batch_labels, Y_train[0:k]])
        
        feed_dict = {x: batch_data,
                     y: batch_labels,
                     K.learning_phase(): 1}

        # choose source of adversarial examples at random
        # (for ensemble adversarial training)
        if x_advs is not None:
            feed_dict[idx] = np.random.randint(len(x_advs))

        # Run the graph
        _, curr_loss, curr_l1, curr_l2, curr_preds, _ = \
            K.get_session().run([optimizer, loss, l1, l2, preds]
                                + [model.updates],
                                feed_dict=feed_dict)

        if step % EVAL_FREQUENCY == 0:
            elapsed_time = time.time() - start_time
            start_time = time.time()
            print('Step %d (epoch %.2f), %.2f s' %
                (step, float(step) * FLAGS.BATCH_SIZE / train_size,
                 elapsed_time))
            print('Minibatch loss: %.3f (%.3f, %.3f)' % (curr_loss, curr_l1, curr_l2))

            print('Minibatch error: %.1f%%' % error_rate(curr_preds, batch_labels))

            sys.stdout.flush()

        step += 1
        if step == num_steps:
            break


def tf_test_error_rate(model, x, X_test, y_test):
    """
    Compute test error.
    """
    assert len(X_test) == len(y_test)

    # Predictions for the test set
    eval_prediction = K.softmax(model(x))

    predictions = batch_eval([x], [eval_prediction], [X_test])[0]

    return error_rate(predictions, y_test)


def error_rate(predictions, labels):
    """
    Return the error rate in percent.
    """

    assert len(predictions) == len(labels)

    return 100.0 - (100.0 * np.sum(np.argmax(predictions, 1) == np.argmax(labels, 1)) / predictions.shape[0])
