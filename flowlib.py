import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import re
import sys

def read_flo(filename):
    with open(filename, 'rb') as f:
        magic = np.fromfile(f, np.float32, count=1)
        if 202021.25 != magic:
            print('Magic number incorrect. Invalid .flo file')
        else:
            w = np.fromfile(f, np.int32, count=1)
            h = np.fromfile(f, np.int32, count=1)
            data = np.fromfile(f, np.float32, count=int(2*w*h))
            # Reshape data into 3D array (columns, rows, bands)
            data2D = np.resize(data, (h[0], w[0],2))
            return data2D    

def write_flo(filename, flow):
    """
    write optical flow in Middlebury .flo format
    :param flow: optical flow map
    :param filename: optical flow file path to be saved
    :return: None
    """
    f = open(filename, 'wb')
    magic = np.array([202021.25], dtype=np.float32)
    (height, width) = flow.shape[0:2]
    w = np.array([width], dtype=np.int32)
    h = np.array([height], dtype=np.int32)
    magic.tofile(f)
    w.tofile(f)
    h.tofile(f)
    flow.tofile(f)
    f.close()


def read_pfm(file):
    file = open(file, 'rb')

    color = None
    width = None
    height = None
    scale = None
    endian = None

    header = file.readline().rstrip()
    header = header.decode('utf-8')
    if header == 'PF':
        color = True
    elif header == 'Pf':
        color = False
    else:
        raise Exception('Not a PFM file.')

    dim_match = re.match(r'^(\d+)\s(\d+)\s$', file.readline().decode('utf-8'))
    if dim_match:
        width, height = map(int, dim_match.groups())
    else:
        raise Exception('Malformed PFM header.')

    scale = float(file.readline().rstrip().decode('utf-8'))
    if scale < 0:  # little-endian
        endian = '<'
        scale = -scale
    else:
        endian = '>'  # big-endian

    data = np.fromfile(file, endian + 'f')
    shape = (height, width, 3) if color else (height, width)

    data = np.reshape(data, shape)
    data = np.flipud(data)
    data = data[:, :, :2]
    return data


def write_pfm(file, image, scale=1):
    file = open(file, 'wb')

    color = None

    if image.dtype.name != 'float32':
        raise Exception('Image dtype must be float32.')

    image = np.flipud(image)

    if len(image.shape) == 3 and image.shape[2] == 3:  # color image
        color = True
    elif len(image.shape) == 2 or len(image.shape) == 3 and image.shape[2] == 1:  # greyscale
        color = False
    else:
        raise Exception('Image must have H x W x 3, H x W x 1 or H x W dimensions.')

    file.write('PF\n' if color else 'Pf\n')
    file.write('%d %d\n' % (image.shape[1], image.shape[0]))

    endian = image.dtype.byteorder

    if endian == '<' or endian == '=' and sys.byteorder == 'little':
        scale = -scale

    file.write('%f\n' % scale)

    image.tofile(file)


def flow_to_color(flow, mask=None, max_flow=None):
    """Converts flow to 3-channel color image.

    Args:
        flow: tensor of shape [num_batch, height, width, 2].
        mask: flow validity mask of shape [num_batch, height, width, 1].
    """
    n = 8
    num_batch, height, width, _ = tf.unstack(tf.shape(flow))
    mask = tf.ones([num_batch, height, width, 1]) if mask is None else mask
    flow_u, flow_v = tf.unstack(flow, axis=3)
    if max_flow is not None:
        max_flow = tf.maximum(tf.to_float(max_flow), 1.)
    else:
        max_flow = tf.reduce_max(tf.abs(flow * mask))
    mag = tf.sqrt(tf.reduce_sum(tf.square(flow), 3))
    angle = tf.atan2(flow_v, flow_u)

    im_h = tf.mod(angle / (2 * np.pi) + 1.0, 1.0)
    im_s = tf.clip_by_value(mag * n / max_flow, 0, 1)
    im_v = tf.clip_by_value(n - im_s, 0, 1)
    im_hsv = tf.stack([im_h, im_s, im_v], 3)
    im = tf.image.hsv_to_rgb(im_hsv)
    return im * mask


def flow_error_image(flow_1, flow_2, mask_occ, mask_noc=None, log_colors=True):
    """Visualize the error between two flows as 3-channel color image.

    Adapted from the KITTI C++ devkit.

    Args:
        flow_1: first flow of shape [num_batch, height, width, 2].
        flow_2: second flow (ground truth)
        mask_occ: flow validity mask of shape [num_batch, height, width, 1].
            Equals 1 at (occluded and non-occluded) valid pixels.
        mask_noc: Is 1 only at valid pixels which are not occluded.
    """
    mask_noc = tf.ones(tf.shape(mask_occ)) if mask_noc is None else mask_noc
    diff_sq = (flow_1 - flow_2) ** 2
    diff = tf.sqrt(tf.reduce_sum(diff_sq, [3], keepdims=True))
    if log_colors:
        num_batch, height, width, _ = tf.unstack(tf.shape(flow_1))
        colormap = [
            [0,0.0625,49,54,149],
            [0.0625,0.125,69,117,180],
            [0.125,0.25,116,173,209],
            [0.25,0.5,171,217,233],
            [0.5,1,224,243,248],
            [1,2,254,224,144],
            [2,4,253,174,97],
            [4,8,244,109,67],
            [8,16,215,48,39],
            [16,1000000000.0,165,0,38]]
        colormap = np.asarray(colormap, dtype=np.float32)
        colormap[:, 2:5] = colormap[:, 2:5] / 255
        mag = tf.sqrt(tf.reduce_sum(tf.square(flow_2), 3, keepdims=True))
        error = tf.minimum(diff / 3, 20 * diff / mag)
        im = tf.zeros([num_batch, height, width, 3])
        for i in range(colormap.shape[0]):
            colors = colormap[i, :]
            cond = tf.logical_and(tf.greater_equal(error, colors[0]),
                                  tf.less(error, colors[1]))
            im = tf.where(tf.tile(cond, [1, 1, 1, 3]),
                           tf.ones([num_batch, height, width, 1]) * colors[2:5],
                           im)
        im = tf.where(tf.tile(tf.cast(mask_noc, tf.bool), [1, 1, 1, 3]),
                       im, im * 0.5)
        im = im * mask_occ
    else:
        error = (tf.minimum(diff, 5) / 5) * mask_occ
        im_r = error # errors in occluded areas will be red
        im_g = error * mask_noc
        im_b = error * mask_noc
        im = tf.concat(axis=3, values=[im_r, im_g, im_b])
    return im