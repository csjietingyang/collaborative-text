import cv2
import numpy as np
from scipy import signal
import tensorflow as tf


# the Residual blocks proposed by Lim et al. in the paper 'Enhanced Deep Residual Networks for Single Image Super-Resolution'
def RSB(input, channels, kernel_size, scale):
    x = tf.layers.conv2d(input, channels, kernel_size, (1, 1), padding='same', activation=None)
    x = tf.nn.relu(x)
    x = tf.layers.conv2d(x, channels, kernel_size, (1, 1), padding='same', activation=None)

    x *= scale

    return input + x


def convolvingImage(im, mask):
    return np.abs(signal.convolve2d(im, mask, mode='same'))


def computingEdge(im):
    if(len(im.shape) == 2):
        im = np.expand_dims(im, axis=2)

    channel = im.shape[2]

    # mask
    x0 = np.array([[1, -1]])
    x1 = np.array([[-1, 1]])
    y0 = np.array([[1], [-1]])
    y1 = np.array([[-1], [1]])

    # convolve
    edge_x0, edge_x1, edge_y0, edge_y1 = 0, 0, 0, 0
    for c in range(channel):
        edge_x0 += convolvingImage(im[:, :, c], x0)
        edge_x1 += convolvingImage(im[:, :, c], x1)
        edge_y0 += convolvingImage(im[:, :, c], y0)
        edge_y1 += convolvingImage(im[:, :, c], y1)

    edge = (edge_x0 + edge_x1 + edge_y0 + edge_y1) / 4.0
    
    edge = edge.astype('uint8')

    return edge


def psnr(im1, im2):
    if(len(im1.shape) == 2 or len(im2.shape) == 2):
        im1 = np.expand_dims(im1, axis=2)
        im2 = np.expand_dims(im2, axis=2)

    im1 = im1.astype(np.float32)
    im2 = im2.astype(np.float32)

    B = 8
    m, n, k = im1.shape
    diff = np.power(im1 - im2, 2)
    MAX = 2**B - 1
    mse = np.sum(diff) / (m * n * k)
    sqrt_mse = np.sqrt(mse)
    mean_psnr = 20 * np.log10(MAX / sqrt_mse)

    return mean_psnr


def ssim(im1, im2):
    k1 = 0.01
    k2 = 0.03
    L = 255
    c1 = (k1 * L) ** 2
    c2 = (k2 * L) ** 2

    im1 = im1.astype(np.float32)
    im2 = im2.astype(np.float32)

    im1_2 = im1 * im1
    im2_2 = im2 * im2
    im1_im2 = im1 * im2

    mu1 = cv2.GaussianBlur(im1, (11, 11), 1.5)
    mu2 = cv2.GaussianBlur(im2, (11, 11), 1.5)

    mu1_2 = mu1 * mu1
    mu2_2 = mu2 * mu2
    mu1_mu2 = mu1 * mu2

    sigma1_2 = cv2.GaussianBlur(im1_2, (11, 11), 1.5)
    sigma1_2 -= mu1_2

    sigma2_2 = cv2.GaussianBlur(im2_2, (11, 11), 1.5)
    sigma2_2 -= mu2_2

    sigma12 = cv2.GaussianBlur(im1_im2, (11, 11), 1.5)
    sigma12 -= mu1_mu2

    t1 = 2 * mu1_mu2 + c1
    t2 = 2 * sigma12 + c2
    t3 = t1 * t2

    t1 = mu1_2 + mu2_2 + c1
    t2 = sigma1_2 + sigma2_2 + c2
    t1 = t1 * t2

    ssim = t3 / t1
    mean_ssim = np.mean(ssim)

    return mean_ssim
