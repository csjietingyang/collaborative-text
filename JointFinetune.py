import argparse
import cv2
import numpy as np
import os
import tensorflow as tf 

import im2tfrecord
from tfrecord2im import tfrecord2im
from utils import RSB, psnr, ssim


class JointFinetune():

    def __init__(self, args):
        print('Building JointFinetune...')

        self.channels = args.channels
        self.up_kernel_size = args.up_kernel_size
        self.pre_kernel_size = args.pre_kernel_size
        self.RSB_scale = args.RSB_scale
        self.RSB_num = args.RSB_num
        self.EPCNN_out_channels = args.EPCNN_out_channels
        self.GRCNN_out_channels = args.GRCNN_out_channels

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


    def EPCNN_inference(self, input):
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

            x = tf.layers.conv2d(x, self.EPCNN_out_channels, self.pre_kernel_size, (1, 1), padding='same')

            return x


    def midst(self, input, edge, size_x, size_y):
        if(self.mode == 'train'):
            batch_size = self.batch_size
        else:
            batch_size = 1

        input_3c = tf.slice(input, [0, 0, 0, 0], [batch_size, size_x, size_y, 3])
        output = tf.concat([input_3c, edge], axis=-1)
        return output

    def GRCNN_inference(self, input, edge):
        with tf.variable_scope('GRCNN'):

            # upsampling module
            x = tf.layers.conv2d_transpose(input, self.channels, self.up_kernel_size, (2, 2), padding='same', activation=tf.nn.relu)

            x = tf.layers.conv2d(x, self.channels, self.up_kernel_size, (1, 1), padding='same', activation=tf.nn.relu)

            x = tf.layers.conv2d_transpose(x, self.channels, self.up_kernel_size, (2, 2), padding='same', activation=tf.nn.relu)

            x = self.midst(x, edge, self.train_size, self.train_size)

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

            x = tf.layers.conv2d(x, self.GRCNN_out_channels, self.pre_kernel_size, (1, 1), padding='same')

            return x


    def EPCNN_loss_conv2d(self, input):
        x_kernel = tf.constant([[1, 0, -1]], dtype=tf.float32, shape=[1, 3, 1, 1])
        y_kernel = tf.constant([[1], [0],[-1]], dtype=tf.float32, shape=[3, 1, 1, 1])

        c = tf.slice(input, [0, 0, 0, 0], [self.batch_size, self.train_size, self.train_size, 1])
        x = tf.nn.conv2d(c, x_kernel, strides=[1, 1, 1, 1], padding='SAME')
        y = tf.nn.conv2d(c, y_kernel, strides=[1, 1, 1, 1], padding='SAME')

        return x, y


    def GRCNN_loss_conv2d(self, input):
        x_kernel = tf.constant([[1, 0, -1]], dtype=tf.float32, shape=[1, 3, 1, 1])
        y_kernel = tf.constant([[1], [0],[-1]], dtype=tf.float32, shape=[3, 1, 1, 1])

        c1 = tf.slice(input, [0, 0, 0, 0], [self.batch_size, self.train_size, self.train_size, 1])
        c2 = tf.slice(input, [0, 0, 0, 1], [self.batch_size, self.train_size, self.train_size, 1])
        c3 = tf.slice(input, [0, 0, 0, 2], [self.batch_size, self.train_size, self.train_size, 1])

        x1 = tf.nn.conv2d(c1, x_kernel, strides=[1, 1, 1, 1], padding='SAME')
        x2 = tf.nn.conv2d(c2, x_kernel, strides=[1, 1, 1, 1], padding='SAME')
        x3 = tf.nn.conv2d(c3, x_kernel, strides=[1, 1, 1, 1], padding='SAME')
        y1 = tf.nn.conv2d(c1, y_kernel, strides=[1, 1, 1, 1], padding='SAME')
        y2 = tf.nn.conv2d(c2, y_kernel, strides=[1, 1, 1, 1], padding='SAME')
        y3 = tf.nn.conv2d(c3, y_kernel, strides=[1, 1, 1, 1], padding='SAME')
        return x1, x2, x3, y1, y2, y3

    def EPCNN_loss(self, input, gt):
        s = tf.reduce_sum(tf.square(gt - input), axis=[1, 2, 3])
        s = 0.8 * s

        input_x, input_y = self.EPCNN_loss_conv2d(input)
        gt_x, gt_y = self.EPCNN_loss_conv2d(gt)
        b = (gt_x - input_x) + (gt_y - input_y)
        l1 = 0.4 * tf.reduce_sum(b, axis=[1, 2, 3])

        l = tf.reduce_mean(s + l1)
        return l


    def GRCNN_loss(self, input, gt):
        s = tf.reduce_sum(tf.square(gt - input), axis=[1, 2, 3])
        s = 0.2 * s

        input_x1, input_x2, input_x3, input_y1, input_y2, input_y3 = self.GRCNN_loss_conv2d(input)
        gt_x1, gt_x2, gt_x3, gt_y1, gt_y2, gt_y3 = self.GRCNN_loss_conv2d(gt)
        b = (gt_x1 - input_x1) + (gt_x2 - input_x2) + (gt_x3 - input_x3) + \
            (gt_y1 - input_y1) + (gt_y2 - input_y2) + (gt_y3 - input_y3)
        l1 = 0.4 * tf.reduce_sum(b, axis=[1, 2, 3])

        l = tf.reduce_mean(s + l1)
        return l


    def loss(self, epcnn_output, grcnn_output, target_edge, target_image):
        epcnn_loss = self.EPCNN_loss(epcnn_output, target_edge)
        grcnn_loss = self.GRCNN_loss(grcnn_output, target_image)
        return 0.4 * epcnn_loss + grcnn_loss


    def optimization(self, loss, lr):
        optimizer = tf.train.AdamOptimizer(lr, name='AdamOptimizer')
        update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)

        with tf.control_dependencies(update_ops):
            train_op = optimizer.minimize(loss)
        return train_op


    def train(self):
        self.mode = 'train'

        parser = argparse.ArgumentParser()
        parser.add_argument("--img_size", default=self.train_size, type=int)
        parser.add_argument("--scale", default=self.sr_scale, type=int)
        parser.add_argument("--tfrecord_path", default='./tfrecord/Xu et al.\'s dataset/')
        parser.add_argument("--save_mode", default='train')
        parser.add_argument("--batch_size", default=self.batch_size, type=int)
        args = parser.parse_args()

        buildImage = tfrecord2im(args)

        with tf.Graph().as_default():
            EPCNN_input = tf.placeholder(shape=[None, self.train_down_size, self.train_down_size, 4], dtype=tf.float32)
            GRCNN_input = tf.placeholder(shape=[None, self.train_down_size, self.train_down_size, 3], dtype=tf.float32)
            Tar_edge = tf.placeholder(shape=[None, self.train_size, self.train_size, 1], dtype=tf.float32)
            Tar_image = tf.placeholder(shape=[None, self.train_size, self.train_size, 3], dtype=tf.float32)
            lr = tf.placeholder(tf.float32, name='learning_rate')

            EPCNN_output = self.EPCNN_inference(EPCNN_input)
            EPCNN_output_clip = tf.clip_by_value(EPCNN_output, 0.0, 255.0)
            GRCNN_output = self.GRCNN_inference(GRCNN_input, EPCNN_output_clip)
            GRCNN_output_clip = tf.clip_by_value(GRCNN_output, 0.0, 255.0)

            tf_ep_in, tf_gr_in, tf_tar_edge, tf_tar_image = buildImage.decodeTfrecord()
            tf_ep_in = tf.cast(tf_ep_in, tf.float32)
            tf_gr_in = tf.cast(tf_gr_in, tf.float32)
            tf_tar_edge = tf.cast(tf_tar_edge, tf.float32)
            tf_tar_image = tf.cast(tf_tar_image, tf.float32)

            loss = self.loss(EPCNN_output, GRCNN_output, Tar_edge, Tar_image)
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
                learning_rates[25:] = learning_rates[0] / 10.0
                learning_rates[40:] = learning_rates[25] / 10.0
                learning_rates[50:] = learning_rates[40] / 10.0
                learning_rates[60:] = learning_rates[50] / 10.0

                batch_num = self.train_num // self.batch_size
                print('tarin_num: ', self.train_num)
                print('batch_num: ', batch_num)
                print('iteration_num: ', batch_num * (self.epochs - epoch_cur))
                for epoch in range(epoch_cur, self.epochs):
                    all_lost = 0
                    for batch in range(0, batch_num):
                        ep_input, gr_input, target_edge, target_image = sess.run([tf_ep_in, tf_gr_in, tf_tar_edge, tf_tar_image])
                        _, lost, summary = sess.run([opt, loss, merged], feed_dict={EPCNN_input: ep_input, GRCNN_input: gr_input, 
                                                                                    Tar_edge: target_edge, Tar_image: target_image, 
                                                                                    lr: learning_rates[epoch]})
                        print('lost: ', lost)
                        target_image = np.squeeze(target_image[0, :, :, :])
                        target_image = target_image.astype('uint8')
                        cv2.imshow('a', target_image)
                        cv2.waitKey(0)
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
        self.mode = 'test'

        test_input_path = './dataset/Xu et al.\'s dataset/TEST/INPUT/'
        test_gt_path = './dataset/Xu et al.\'s dataset/TEST/GT/'
        save_path = './dataset/Xu et al.\'s dataset/M0/JointFinetune/'
        if not os.path.exists(save_path):
            os.mkdir(save_path)

        test_input_list = [im for im in os.listdir(test_input_path) if im.endswith('.png')]
        test_gt_list = [im for im in os.listdir(test_gt_path) if im.endswith('.png')]
        
        test_num = len(test_input_list)
        print('Num. of test patches: ', test_num)

        edge_psnr_file = np.zeros(test_num)
        edge_ssim_file = np.zeros(test_num)
        psnr_file = np.zeros(test_num)
        ssim_file = np.zeros(test_num)

        test_size = 200
        test_down_size = test_size // self.sr_scale

        with tf.Graph().as_default():
            EPCNN_input = tf.placeholder(shape=[None, self.train_down_size, self.train_down_size, 4], dtype=tf.float32)
            GRCNN_input = tf.placeholder(shape=[None, self.train_down_size, self.train_down_size, 3], dtype=tf.float32)
            Tar_edge = tf.placeholder(shape=[None, self.train_size, self.train_size, 1], dtype=tf.float32)
            Tar_image = tf.placeholder(shape=[None, self.train_size, self.train_size, 3], dtype=tf.float32)

            EPCNN_output = self.EPCNN_inference(EPCNN_input)
            EPCNN_output = tf.clip_by_value(EPCNN_output, 0.0, 255.0)
            GRCNN_output = self.GRCNN_inference(GRCNN_input, EPCNN_output)
            GRCNN_output = tf.clip_by_value(GRCNN_output, 0.0, 255.0)

            para_num = np.sum([np.prod(v.get_shape().as_list()) for v in tf.trainable_variables()])
            print('Num. of Parameters: ', para_num)

            var_list = [v for v in tf.all_variables() if v.name.startswith('EPCNN') or v.name.startswith('GRCNN')]
            saver = tf.train.Saver(var_list)

            with tf.Session() as sess:
                saver.restore(sess, os.path.join(self.model_path, model))

                for i in range(test_num):
                    ep_input, gr_input, target_edge, target_image = im2tfrecord.generatingSyntheticEdge(os.path.join(test_input_path, test_input_list[i]),
                                                                                                        os.path.join(test_gt_path, test_gt_list[i]))
                    ep_input = ep_input.astype(np.float32)
                    gr_input = gr_input.astype(np.float32)
                    target_edge = target_edge.astype(np.float32)
                    target_image = target_image.astype(np.float32)
                    ep_input = np.expand_dims(ep_input, axis=0)
                    gr_input = np.expand_dims(gr_input, axis=0)
                    target_edge = np.expand_dims(target_edge, axis=0)
                    target_edge = np.expand_dims(target_edge, axis=3)
                    target_image = np.expand_dims(target_image, axis=0)

                    ep_output, gr_output = sess.run([EPCNN_output, GRCNN_output], feed_dict={EPCNN_input: ep_input, GRCNN_input: gr_input, 
                                                               Tar_edge: target_edge, Tar_image: target_image})
                    ep_output = np.squeeze(ep_output)
                    gr_output = np.squeeze(gr_output)
                    target_edge = np.squeeze(target_edge)
                    target_image = np.squeeze(target_image)
                    
                    ep_output = ep_output.astype('uint8')
                    gr_output = gr_output.astype('uint8')
                    target_edge = target_edge.astype('uint8')
                    target_image = target_image.astype('uint8')

                    edge_psnr_file[i] = psnr(ep_output, target_edge)
                    edge_ssim_file[i] = ssim(ep_output, target_edge)
                    psnr_file[i] = psnr(gr_output, target_image)
                    ssim_file[i] = ssim(gr_output, target_image)

                    save_name = test_input_list[i].split('.')[0][:-5]
                    cv2.imwrite(os.path.join(save_path, save_name + '_output_edge.png'), ep_output)
                    cv2.imwrite(os.path.join(save_path, save_name + '_output.png'), gr_output)

                print('JointFinetune: ', model)
                print('Edge PSNR: ', str(np.mean(edge_psnr_file)))
                print('Edge SSIM: ', str(np.mean(edge_ssim_file)))
                print('PSNR: ', str(np.mean(psnr_file)))
                print('SSIM: ', str(np.mean(ssim_file)))


if __name__ == '__main__':
    m_path = './model/'
    if not os.path.isdir(m_path):
        os.mkdir(m_path)

    m_path = './model/Xu et al.\'s dataset/'
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
    parser.add_argument("--EPCNN_out_channels", default=1, type=int)
    parser.add_argument("--GRCNN_out_channels", default=3, type=int)
    parser.add_argument("--train_size", default=200, type=int)
    parser.add_argument("--sr_scale", default=4, type=int)
    parser.add_argument("--train_num", default=1046368, type=int)
    parser.add_argument("--batch_size", default=16, type=int)
    parser.add_argument("--learning_rate", default=0.0001, type=float)
    parser.add_argument("--model_path", default='./model/Xu et al.\'s dataset/JointFinetune/')
    parser.add_argument("--epochs", default=62, type=int)
    args = parser.parse_args()

    joint_finetune_network = JointFinetune(args)
    joint_finetune_network.train()
    joint_finetune_network.test(model='model_weight_62.ckpt')
