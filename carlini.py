## li_attack.py -- attack a network optimizing for l_infinity distance
##
## Adapted from https://github.com/carlini/nn_robust_attacks
##
## Copyright (C) 2016, Nicholas Carlini <nicholas@carlini.com>.
##
## This program is licenced under the BSD 2-Clause licence,
## contained in the LICENCE file in this directory.

import tensorflow as tf
import numpy as np
from tensorflow.python.platform import flags
import keras.backend as K

MAX_ITERATIONS = 1000   # number of iterations to perform gradient descent
ABORT_EARLY = True      # abort gradient descent upon first valid solution
INITIAL_CONST = 1e-3    # the first value of c to start at
LEARNING_RATE = 5e-3    # larger values converge faster to less accurate results
LARGEST_CONST = 2e+1    # the largest value of c to go up to before giving up
TARGETED = True         # should we target one specific class? or just be wrong?
CONST_FACTOR = 10.0     # f>1, rate at which we increase constant, smaller better
CONFIDENCE = 0          # how strong the adversarial example should be
EPS = 0.3

FLAGS = flags.FLAGS


class CarliniLi:
    def __init__(self, sess, model,
                 targeted = TARGETED, learning_rate = LEARNING_RATE,
                 max_iterations = MAX_ITERATIONS, abort_early = ABORT_EARLY,
                 initial_const = INITIAL_CONST, largest_const = LARGEST_CONST,
                 const_factor = CONST_FACTOR, confidence = CONFIDENCE, eps=EPS):
        """
        The L_infinity optimized attack.
        Returns adversarial examples for the supplied model.
        targeted: True if we should perform a targetted attack, False otherwise.
        learning_rate: The learning rate for the attack algorithm. Smaller values
          produce better results but are slower to converge.
        max_iterations: The maximum number of iterations. Larger values are more
          accurate; setting too small will require a large learning rate and will
          produce poor results.
        abort_early: If true, allows early aborts if gradient descent gets stuck.
        initial_const: The initial tradeoff-constant to use to tune the relative
          importance of distance and confidence. Should be set to a very small
          value (but positive).
        largest_const: The largest constant to use until we report failure. Should
          be set to a very large value.
        reduce_const: If true, after each successful attack, make const smaller.
        decrease_factor: Rate at which we should decrease tau, less than one.
          Larger produces better quality results.
        const_factor: The rate at which we should increase the constant, when the
          previous constant failed. Should be greater than one, smaller is better.
        """
        self.model = model
        self.sess = sess

        self.TARGETED = targeted
        self.LEARNING_RATE = learning_rate
        self.MAX_ITERATIONS = max_iterations
        self.ABORT_EARLY = abort_early
        self.INITIAL_CONST = initial_const
        self.LARGEST_CONST = largest_const
        self.const_factor = const_factor
        self.CONFIDENCE = confidence
        self.EPS = eps

        self.grad = self.gradient_descent(sess, model)

    def gradient_descent(self, sess, model):
        def compare(outputs, labels):
            y = np.argmax(labels)
            pred = np.argmax(outputs)

            if self.TARGETED:
                return (pred == y)
            else:
                return (pred != y)

        shape = (1, FLAGS.IMAGE_ROWS, FLAGS.IMAGE_COLS, FLAGS.NUM_CHANNELS)

        # the variable to optimize over
        modifier = tf.Variable(np.zeros(shape,dtype=np.float32))

        tau = tf.placeholder(tf.float32, [])
        simg = tf.placeholder(tf.float32, shape)
        timg = tf.placeholder(tf.float32, shape)
        tlab = tf.placeholder(tf.float32, (1, FLAGS.NUM_CLASSES))
        const = tf.placeholder(tf.float32, [])

        newimg = tf.clip_by_value(simg + modifier, 0, 1)

        output = model(newimg)
        orig_output = model(timg)

        real = tf.reduce_sum((tlab)*output)
        other = tf.reduce_max((1-tlab)*output - (tlab*10000))

        if self.TARGETED:
            # if targetted, optimize for making the other class most likely
            loss1 = tf.maximum(0.0,other-real+self.CONFIDENCE)
        else:
            # if untargeted, optimize for making this class least likely.
            loss1 = tf.maximum(0.0,real-other+self.CONFIDENCE)

        # sum up the losses
        loss2 = tf.reduce_sum(tf.maximum(0.0, tf.abs(newimg-timg)-tau))
        loss = const*loss1+loss2

        # setup the adam optimizer and keep track of variables we're creating
        start_vars = set(x.name for x in tf.global_variables())
        optimizer = tf.train.AdamOptimizer(self.LEARNING_RATE)
        train = optimizer.minimize(loss, var_list=[modifier])

        end_vars = tf.global_variables()
        new_vars = [x for x in end_vars if x.name not in start_vars]
        init = tf.variables_initializer(var_list=[modifier]+new_vars)

        def doit(oimgs, labs, starts, tt, CONST):
            prev_scores = None

            imgs = np.array(oimgs)
            starts = np.array(starts)

            # initialize the variables
            sess.run(init)
            while CONST < self.LARGEST_CONST:
                # try solving for each value of the constant
                #print('try const', CONST)
                for step in range(self.MAX_ITERATIONS):
                    feed_dict={timg: imgs,
                               tlab:labs,
                               tau: tt,
                               simg: starts,
                               const: CONST,
                               K.learning_phase(): 0}

                    #if step % (self.MAX_ITERATIONS//10) == 0:
                    #    print(step, sess.run((loss,loss1,loss2),feed_dict=feed_dict))

                    # perform the update step
                    _, works, linf_slack = sess.run([train, loss, loss2], feed_dict=feed_dict)

                    # it worked
                    if works < .0001*CONST and (self.ABORT_EARLY or step == CONST-1):
                        get = sess.run(K.softmax(output), feed_dict=feed_dict)
                        works = compare(get, labs)
                        if works:
                            scores, origscores, nimg = sess.run((output,orig_output,newimg),feed_dict=feed_dict)
                            return scores, origscores, nimg, CONST

                # we didn't succeed, increase constant and try again

                if linf_slack >= 0.1 * self.EPS:
                    # perturbation is too large
                    if prev_scores is None:
                        return prev_scores
                    return prev_scores, prev_origscores, prev_nimg, CONST
                else:
                    # didn't reach target confidence
                    CONST *= self.const_factor

                prev_scores, prev_origscores, prev_nimg = sess.run((output,orig_output,newimg),feed_dict=feed_dict)

            scores, origscores, nimg = sess.run((output,orig_output,newimg),feed_dict=feed_dict)
            return scores, origscores, nimg, CONST

        return doit

    def attack(self, imgs, targets):
        """
        Perform the L_0 attack on the given images for the given targets.
        If self.targeted is true, then the targets represents the target labels.
        If self.targeted is false, then targets are the original class labels.
        """
        r = []
        i = 0
        for img,target in zip(imgs, targets):
            print i
            r.extend(self.attack_single(img, target))
            i += 1
        return np.array(r)

    def attack_single(self, img, target):
        """
        Run the attack on a single image and label
        """

        # the previous image
        prev = np.copy(img).reshape((1, FLAGS.IMAGE_ROWS, FLAGS.IMAGE_COLS, FLAGS.NUM_CHANNELS))
        tau = self.EPS
        const = self.INITIAL_CONST

        res = self.grad([np.copy(img)], [target], np.copy(prev), tau, const)

        if res is None:
            # the attack failed, we return this as our final answer
            return prev

        scores, origscores, nimg, const = res
        prev = nimg
        return prev
