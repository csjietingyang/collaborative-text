import argparse
import cv2
import numpy as np
import os
import tensorflow as tf 

import im2tfrecord
from tfrecord2im import tfrecord2im
from utils import RSB, psnr, ssim


class EPCNN():

    def __init__(self, args):
        print('Building EPCNN...')

        self.channels = args.channels
        self.up_kernel_size = args.up_kernel_size
        self.pre_kernel_size = args.pre_kernel_size
        self.RSB_scale = args.RSB_scale
        self.RSB_num = args.RSB_num
        self.out_channels = args.out_channels

        self.train_size = args.train_size
        self.sr_scale = args.sr_scale
        self.train_down_size = self.train_size // self.sr_scale
        self.train_num = args.train_num
        self.batch_size = args.batch_size
        self.learning_rate = args.learning_rate
        self.model_path = args.model_path
        self.summary_dir = os.path.join(self.model_path, 'logs')
        self.epochs = args.epochs

        if not os.path.isdir(self.model_path):
            os.mkdir(self.model_path)
            os.mkdir(self.summary_dir)

        print('Done Building!')


    def inference(self, input):
        with tf.variable_scope('EPCNN'):

            # upsampling module
            x = tf.layers.conv2d_transpose(input, self.channels, self.up_kernel_size, (2, 2), padding='same', activation=tf.nn.relu)

            x = tf.layers.conv2d(x, self.channels, self.up_kernel_size, (1, 1), padding='same', activation=tf.nn.relu)

            x = tf.layers.conv2d_transpose(x, self.channels, self.up_kernel_size, (2, 2), padding='same', activation=tf.nn.relu)

            # prediction module
            x = tf.layers.conv2d(x, self.channels, self.pre_kernel_size, (1, 1), padding='same', activation=tf.nn.relu)

            x = tf.layers.conv2d(x, self.channels, self.pre_kernel_size, (1, 1), padding='same', activation=tf.nn.relu)

            x = tf.layers.conv2d(x, self.channels, self.pre_kernel_size, (2, 2), padding='same', activation=tf.nn.relu)

            x_before = 1 * x

            for i in range(self.RSB_num):
                x = RSB(x, self.channels, self.pre_kernel_size, self.RSB_scale)

            x_after = 1 * x

            x = tf.add(x_before, x_after)

            x = tf.layers.conv2d_transpose(x, self.channels, self.pre_kernel_size, (2, 2), padding='same', activation=tf.nn.relu)

            x = tf.layers.conv2d(x, self.channels, self.pre_kernel_size, (1, 1), padding='same', activation=tf.nn.relu)

            x = tf.layers.conv2d(x, self.out_channels, self.pre_kernel_size, (1, 1), padding='same')

            return x


    def loss_conv2d(self, input):
        x_kernel = tf.constant([[1, 0, -1]], dtype=tf.float32, shape=[1, 3, 1, 1])
        y_kernel = tf.constant([[1], [0],[-1]], dtype=tf.float32, shape=[3, 1, 1, 1])

        c = tf.slice(input, [0, 0, 0, 0], [self.batch_size, self.train_size, self.train_size, 1])
        x = tf.nn.conv2d(c, x_kernel, strides=[1, 1, 1, 1], padding='SAME')
        y = tf.nn.conv2d(c, y_kernel, strides=[1, 1, 1, 1], padding='SAME')

        return x, y


    def loss(self, input, gt):
        s = tf.reduce_sum(tf.square(gt - input), axis=[1, 2, 3])
        s = 0.8 * s

        input_x, input_y = self.loss_conv2d(input)
        gt_x, gt_y = self.loss_conv2d(gt)
        b = (gt_x - input_x) + (gt_y - input_y)
        l1 = 0.4 * tf.reduce_sum(b, axis=[1, 2, 3])

        l = tf.reduce_mean(s + l1)
        return l


    def optimization(self, loss, lr):
        optimizer = tf.train.AdamOptimizer(lr, name='AdamOptimizer')
        update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)

        with tf.control_dependencies(update_ops):
            train_op = optimizer.minimize(loss)
        return train_op


    def train(self):
        parser = argparse.ArgumentParser()
        parser.add_argument("--img_size", default=self.train_size, type=int)
        parser.add_argument("--scale", default=self.sr_scale, type=int)
        parser.add_argument("--tfrecord_path", default='./tfrecord/ComTex/')
        parser.add_argument("--save_mode", default='train')
        parser.add_argument("--batch_size", default=self.batch_size, type=int)
        args = parser.parse_args()

        buildImage = tfrecord2im(args)

        with tf.Graph().as_default():
            EPCNN_input = tf.placeholder(shape=[None, self.train_down_size, self.train_down_size, 4], dtype=tf.float32)
            Tar_edge = tf.placeholder(shape=[None, self.train_size, self.train_size, 1], dtype=tf.float32)
            lr = tf.placeholder(tf.float32, name='learning_rate')

            EPCNN_output = self.inference(EPCNN_input)
            EPCNN_output_clip = tf.clip_by_value(EPCNN_output, 0.0, 255.0)

            tf_ep_in, _, tf_tar_edge, _ = buildImage.decodeTfrecord()
            tf_ep_in = tf.cast(tf_ep_in, tf.float32)
            tf_tar_edge = tf.cast(tf_tar_edge, tf.float32)

            loss = self.loss(EPCNN_output, Tar_edge)
            opt = self.optimization(loss, lr)

            tf.summary.scalar('loss', loss)
            merged = tf.summary.merge_all()

            init = tf.global_variables_initializer()
            saver = tf.train.Saver(max_to_keep=100)

            with tf.Session() as sess:
                ckpt = tf.train.get_checkpoint_state(self.model_path)
                if ckpt and ckpt.model_checkpoint_path:
                    full_path = tf.train.latest_checkpoint(self.model_path)
                    epoch_cur = int(os.path.basename(full_path).split('.')[0].split('_')[-1])
                    step = epoch_cur * num_example
                    saver.restore(sess, full_path)
                    print("Loading " + os.path.basename(full_path) + " to the model")
                else:
                    sess.run(init)
                    epoch_cur = 0
                    step = 0
                    print("Initialing the model")

                summary_writer = tf.summary.FileWriter(logdir=self.summary_dir, graph=sess.graph)

                coord = tf.train.Coordinator()
                threads = tf.train.start_queue_runners(sess=sess, coord=coord)

                learning_rates = self.learning_rate * np.ones([self.epochs])
                learning_rates[20:] = learning_rates[0] / 10.0

                batch_num = self.train_num // self.batch_size
                print('tarin_num: ', self.train_num)
                print('batch_num: ', batch_num)
                print('iteration_num: ', batch_num * (self.epochs - epoch_cur))
                for epoch in range(epoch_cur, self.epochs):
                    all_lost = 0
                    for batch in range(0, batch_num):
                        ep_input, target_edge = sess.run([tf_ep_in, tf_tar_edge])
                        _, lost, summary = sess.run([opt, loss, merged], feed_dict={EPCNN_input: ep_input, Tar_edge: target_edge, lr: learning_rates[epoch]})
                        all_lost += lost
                        step = batch_num * epoch + batch
                        summary_writer.add_summary(summary, global_step=step)

                    mean_lost = all_lost / batch_num
                    print('----------epoch', str(epoch + 1), '----------')
                    print('loss: ', mean_lost)
                    saver.save(sess, os.path.join(self.model_path, 'model_weight_' + str(epoch + 1) + '.ckpt'))
                    print('----------epoch ', str(epoch + 1), 'is trained successfully +++++')

                summary_writer.close()
                coord.request_stop()
                coord.join(threads)


    def test(self, model):
        test_input_path = './dataset/ComTex/TEST/INPUT/'
        test_gt_path = './dataset/ComTex/TEST/GT/'
        save_path = './dataset/ComTex/M0/EPCNN/'
        if not os.path.exists(save_path):
            os.mkdir(save_path)

        test_input_list = [im for im in os.listdir(test_input_path) if im.endswith('.png')]
        test_gt_list = [im for im in os.listdir(test_gt_path) if im.endswith('.png')]
        
        test_num = len(test_input_list)
        print('Num. of test patches: ', test_num)

        psnr_file = np.zeros(test_num)
        ssim_file = np.zeros(test_num)

        test_size = 200
        test_down_size = test_size // self.sr_scale

        with tf.Graph().as_default():
            EPCNN_input = tf.placeholder(shape=[None, test_down_size, test_down_size, 4], dtype=tf.float32)
            Tar_edge = tf.placeholder(shape=[None, test_size, test_size, 1], dtype=tf.float32)

            EPCNN_output = self.inference(EPCNN_input)
            EPCNN_output = tf.clip_by_value(EPCNN_output, 0.0, 255.0)

            para_num = np.sum([np.prod(v.get_shape().as_list()) for v in tf.trainable_variables()])
            print('Num. of Parameters: ', para_num)

            var_list = [v for v in tf.all_variables() if v.name.startswith('EPCNN')]
            saver = tf.train.Saver(var_list)

            with tf.Session() as sess:
                saver.restore(sess, os.path.join(self.model_path, model))

                for i in range(test_num):
                    ep_input, _, target_edge, _ = im2tfrecord.generatingSyntheticEdge(os.path.join(test_input_path, test_input_list[i]),
                                                                                      os.path.join(test_gt_path, test_gt_list[i]))
                    ep_input = ep_input.astype(np.float32)
                    target_edge = target_edge.astype(np.float32)
                    ep_input = np.expand_dims(ep_input, axis=0)
                    target_edge = np.expand_dims(target_edge, axis=0)
                    target_edge = np.expand_dims(target_edge, axis=3)

                    output = sess.run(EPCNN_output, feed_dict={EPCNN_input: ep_input, Tar_edge: target_edge})
                    output = np.squeeze(output)
                    target_edge = np.squeeze(target_edge)
                    output = output.astype('uint8')
                    target_edge = target_edge.astype('uint8')

                    psnr_file[i] = psnr(output, target_edge)
                    ssim_file[i] = ssim(output, target_edge)

                    save_name = test_input_list[i].split('.')[0][:-5]
                    cv2.imwrite(os.path.join(save_path, save_name + '_output_edge.png'), output)

                print('EPCNN: ', model)
                print('Edge PSNR: ', str(np.mean(psnr_file)))
                print('Edge SSIM: ', str(np.mean(ssim_file)))


if __name__ == '__main__':
    m_path = './model/'
    if not os.path.isdir(m_path):
        os.mkdir(m_path)

    m_path = './model/ComTex/'
    if not os.path.isdir(m_path):
        os.mkdir(m_path)

    m_path = './model/ComTex/'
    if not os.path.isdir(m_path):
        os.mkdir(m_path)


    parser = argparse.ArgumentParser()
    parser.add_argument("--channels", default=64, type=int)
    parser.add_argument("--up_kernel_size", default=6, type=int)
    parser.add_argument("--pre_kernel_size", default=3, type=int)
    parser.add_argument("--RSB_scale", default=0.1, type=int)
    parser.add_argument("--RSB_num", default=13, type=int)
    parser.add_argument("--out_channels", default=1, type=int)
    parser.add_argument("--train_size", default=200, type=int)
    parser.add_argument("--sr_scale", default=4, type=int)
    parser.add_argument("--train_num", default=1046368, type=int)
    parser.add_argument("--batch_size", default=32, type=int)
    parser.add_argument("--learning_rate", default=0.0001, type=float)
    parser.add_argument("--model_path", default='./model/ComTex/EPCNN/')
    parser.add_argument("--epochs", default=25, type=int)
    args = parser.parse_args()

    EPCNN_network = EPCNN(args)
    EPCNN_network.train()
    EPCNN_network.test(model='model_weight_25.ckpt')
