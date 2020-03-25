import os
import tensorflow as tf


class tfrecord2im():

    def __init__(self, args):
        print('Building tfrecord2im...')

        self.img_size = args.img_size
        self.scale = args.scale
        self.down_size = self.img_size // self.scale
        self.tfrecord_path = args.tfrecord_path
        self.save_mode = args.save_mode
        self.tf_name = os.path.join(self.tfrecord_path, self.save_mode + '.tfrecords')
        self.batch_size = args.batch_size

        print('Done building!')
        

    def _generate_image_batch_noshuffle(self, epcnn_input, grcnn_input, target_edge, target_image, min_queue_examples, batch_size):
        num_preprocess_threads = 2
        epcnn_input, grcnn_input, target_edge, target_image = tf.train.shuffle_batch(
            [epcnn_input, grcnn_input, target_edge, target_image],
            batch_size=batch_size,
            num_threads=num_preprocess_threads,
            capacity=min_queue_examples + 3 * batch_size,
            min_after_dequeue = 10)

        return epcnn_input, grcnn_input, target_edge, target_image


    def decodeTfrecord(self):
        # generate a queue with filenames
        filename_queue = tf.train.string_input_producer([self.tf_name])

        # define a reader to read image data from tfrecod files
        reader = tf.TFRecordReader()

        # return filename and file
        _, serialized_example = reader.read(filename_queue)

        # features saving the image information, including raw_data, height, width and channel
        features = tf.parse_single_example(serialized_example, features={
                                        'epcnn_input': tf.FixedLenFeature([], tf.string),
                                        'grcnn_input': tf.FixedLenFeature([], tf.string),
                                        'target_edge': tf.FixedLenFeature([], tf.string),
                                        'target_image': tf.FixedLenFeature([], tf.string)
                                        })

        # decode image and reshape based height, width and channel
        # epcnn_input
        epcnn_input = tf.decode_raw(features['epcnn_input'], tf.uint8)
        epcnn_input = tf.reshape(epcnn_input, [self.down_size, self.down_size, 4])
        epcnn_input = tf.cast(epcnn_input, tf.float32)

        # grcnn_input
        grcnn_input = tf.decode_raw(features['grcnn_input'], tf.uint8)
        grcnn_input = tf.reshape(grcnn_input, [self.down_size, self.down_size, 3])
        grcnn_input = tf.cast(grcnn_input, tf.float32)

        # target_edge
        target_edge = tf.decode_raw(features['target_edge'], tf.uint8)
        target_edge = tf.reshape(target_edge, [self.img_size, self.img_size, 1])
        target_edge = tf.cast(target_edge, tf.float32)

        # target_image
        target_image = tf.decode_raw(features['target_image'], tf.uint8)
        target_image = tf.reshape(target_image, [self.img_size, self.img_size, 3])
        target_image = tf.cast(target_image, tf.float32)

        return self._generate_image_batch_noshuffle(epcnn_input, grcnn_input, target_edge, target_image, 60, self.batch_size)
