import argparse
import cv2
import numpy as np
import os
import tensorflow as tf

from utils import computingEdge


def generatingSyntheticEdge(input, gt):
    input_im = cv2.imread(input)
    gt_im = cv2.imread(gt)

    input_edge = computingEdge(input_im)
    target_edge = computingEdge(gt_im)

    (B, G, R) = cv2.split(input_im)
    epcnn_input = cv2.merge([B, G, R, input_edge])

    return epcnn_input, input_im, target_edge, gt_im


def generatingRealEdge(input):
    input_im = cv2.imread(input)

    input_edge = computingEdge(input_im)

    (B, G, R) = cv2.split(input_im)
    epcnn_input = cv2.merge([B, G, R, input_edge])

    return epcnn_input, input_im



class im2tfrecord():

    def __init__(self, args):
        print('Building im2tfrecord...')

        self.img_size = args.img_size
        self.scale = args.scale
        self.down_size = self.img_size // self.scale
        self.train_input_path = args.train_input_path
        self.train_gt_path = args.train_gt_path
        self.tfrecord_path = args.tfrecord_path
        self.save_mode = args.save_mode
        self.tf_name = os.path.join(self.tfrecord_path, self.save_mode + '.tfrecords')

        if not os.path.isdir(self.tfrecord_path):
            os.mkdir(self.tfrecord_path)

        print('Done building!')


    def writingTfrecord(self):
        # writer for save image into tfrecord
        writer = tf.python_io.TFRecordWriter(self.tf_name)

        # training images
        train_input_list = [im for im in os.listdir(self.train_input_path) if im.endswith('.png')]
        train_gt_list = [im for im in os.listdir(self.train_gt_path) if im.endswith('.png')]
        
        train_num = len(train_input_list)
        print('Num. of training patches: ', train_num)
        print('Begin writing...')

        for i in range(train_num):
            epcnn_input, grcnn_input, target_edge, target_image = generatingSyntheticEdge(os.path.join(self.train_input_path, train_input_list[i]), 
                                                                                          os.path.join(self.train_gt_path, train_gt_list[i]))
            # image to bytes
            ep_raw = epcnn_input.tobytes()
            gr_raw = grcnn_input.tobytes()
            te_raw = target_edge.tobytes()
            ti_raw = target_image.tobytes()

            example = tf.train.Example(features=tf.train.Features(feature={
                'epcnn_input': tf.train.Feature(bytes_list=tf.train.BytesList(value=[ep_raw])),
                'grcnn_input': tf.train.Feature(bytes_list=tf.train.BytesList(value=[gr_raw])),
                'target_edge': tf.train.Feature(bytes_list=tf.train.BytesList(value=[te_raw])),
                'target_image': tf.train.Feature(bytes_list=tf.train.BytesList(value=[ti_raw]))
                
            }))

            # write
            writer.write(example.SerializeToString())

        writer.close()
        print('Done writing!')
    

if __name__ == '__main__':
    tf_path = './tfrecord/'
    if not os.path.isdir(tf_path):
        os.mkdir(tf_path)

    parser = argparse.ArgumentParser()
    parser.add_argument("--img_size", default=200, type=int)
    parser.add_argument("--scale", default=4, type=int)
    parser.add_argument("--train_input_path", default='./dataset/Xu et al.\'s dataset/TRAINING/INPUT/')
    parser.add_argument("--train_gt_path", default='./dataset/Xu et al.\'s dataset/TRAINING/GT/')
    parser.add_argument("--tfrecord_path", default='./tfrecord/Xu et al.\'s dataset/')
    parser.add_argument("--save_mode", default='train')
    args = parser.parse_args()

    buildTfrecord = im2tfrecord(args)
    buildTfrecord.writingTfrecord()
