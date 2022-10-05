""""
If you use this code, you must cite this paper:

Bardozzo, Francesco, et al. "StaSiS-Net: A stacked and siamese disparity estimation network for depth reconstruction 
in modern 3D laparoscopy." Medical Image Analysis 77 (2022): 102380.

Bardozzo, Francesco, et al. "Cross X-AI: Explainable Semantic Segmentation of Laparoscopic Images in Relation 
to Depth Estimation." 2022 International Joint Conference on Neural Networks (IJCNN). IEEE, 2022.

Download the files here: https://drive.google.com/drive/folders/1_atwJnYU61aGYjrKrhh8s32mgfpzYdhh?usp=sharing

> python3 run_stasis.py

Thank you!
"""



from __future__ import absolute_import
from __future__ import print_function
from tensorflow.compat.v1 import ConfigProto
from tensorflow.compat.v1 import InteractiveSession
config = ConfigProto()
config.gpu_options.allow_growth = True
session = InteractiveSession(config=config)

import tensorflow as tf
#Created TensorFlow device (/job:localhost/replica:0/task:0/device:GPU:0 with 5300 MB memory) -
# > physical GPU (device: 0, name: GeForce RTX 2060, pci bus id: 0000:01:00.0, compute capability: 7.5)
# tf version 2.2.0



import os
import argparse

import time
import matplotlib.pyplot as plt
import matplotlib.image as mpimg


parser = argparse.ArgumentParser(description='Stasis-Net - Disparity Estimation Network')
parser.add_argument('--input_height',              type=int,   help='input height', default=192)
parser.add_argument('--input_width',               type=int,   help='input width', default=384)

parser.add_argument('--out_dir',                   type=str, help='save_predictions', default="./run_stasis/output/")
parser.add_argument('--input_dir_left',            type=str, help='input dir L image', default="./run_stasis/input/L/")
parser.add_argument('--input_dir_right',           type=str, help='input dir R image', default="./run_stasis/input/R/")

parser.add_argument('--alpha1',                    type=float, help='alpha1', default=0.1)
parser.add_argument('--alpha2',                    type=float, help='alpha2', default=0.4)
parser.add_argument('--alpha3',                    type=float, help='alpha3', default=0.5)

parser.add_argument('--gpus', type=int, default=1, help='The number of GPUs to use')
parser.add_argument('--gpuids', type=str, default='0', help='IDs of GPUs to use 0,1,2,3,4,5,6,7')

args = parser.parse_args()

# Inform about multi-gpu training
if args.gpus == 1:
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpuids
    print('Will use GPU ' + args.gpuids) #usage 0,1,2,3,4,5,6,7
else:
    print('Will use ' + str(args.gpus) + ' GPUs.')
# Create a MirroredStrategy.
strategy = tf.distribute.MirroredStrategy()
print('Number of devices: {}'.format(strategy.num_replicas_in_sync))



IMG_HEIGHT = args.input_height
IMG_WIDTH  = args.input_width

alpha1  =  args.alpha1
alpha2  =  args.alpha2
alpha3  =  args.alpha3
dir_left  = args.input_dir_left
dir_right = args.input_dir_right

left_imgs  = sorted(os.listdir(dir_left))
right_imgs = sorted(os.listdir(dir_right))




def ssi_loss(y_true, y_pred):
 min_min = tf.reduce_min(tf.stack([tf.reduce_min(y_true), tf.reduce_min(y_pred)], 0))
 max_max = tf.reduce_max(tf.stack([tf.reduce_max(y_true), tf.reduce_max(y_pred)], 0))
 sub2 = tf.subtract(max_max, min_min)
 return   1.0 - tf.reduce_mean(tf.image.ssim(y_true, y_pred, max_val=sub2, filter_size=2, filter_sigma=1.5, k1=0.01, k2=0.03))

def ssi_acc(y_true, y_pred):
 min_min = tf.reduce_min(tf.stack([tf.reduce_min(y_true), tf.reduce_min(y_pred)], 0))
 max_max = tf.reduce_max(tf.stack([tf.reduce_max(y_true), tf.reduce_max(y_pred)], 0))
 sub2 = tf.subtract(max_max, min_min)
 return  tf.reduce_mean(tf.image.ssim(y_true, y_pred, max_val=sub2, filter_size=2, filter_sigma=1.5, k1=0.01, k2=0.03))

def ssi_loss_ms(y_true, y_pred):
 min_min = tf.reduce_min(tf.stack([tf.reduce_min(y_true), tf.reduce_min(y_pred)], 0))
 max_max = tf.reduce_max(tf.stack([tf.reduce_max(y_true), tf.reduce_max(y_pred)], 0))
 sub2 = tf.subtract(max_max, min_min)
 return 1.0 - tf.reduce_mean(tf.image.ssim_multiscale(y_true, y_pred, max_val=sub2, filter_size=2, power_factors=[(0.0448, 0.2856, 0.3001, 0.2363, 0.1333)] ))

def ssi_acc_ms(y_true, y_pred):
 min_min = tf.reduce_min(tf.stack([tf.reduce_min(y_true), tf.reduce_min(y_pred)], 0))
 max_max = tf.reduce_max(tf.stack([tf.reduce_max(y_true), tf.reduce_max(y_pred)], 0))
 sub2 = tf.subtract(max_max, min_min)
 return tf.reduce_mean(tf.image.ssim_multiscale(y_true, y_pred, max_val=sub2, filter_size=2, power_factors=[(0.0448, 0.2856, 0.3001, 0.2363, 0.1333)] ))


def loss_smooth(y_true, y_pred):
    dy_true, dx_true = tf.image.image_gradients(y_true)
    dy_pred, dx_pred = tf.image.image_gradients(y_pred)
    l_edges = tf.reduce_mean(tf.abs(dy_pred - dy_true) + tf.abs(dx_pred - dx_true), axis=-1)
    return l_edges

def acc_smooth(y_true, y_pred):
    dy_true, dx_true = tf.image.image_gradients(y_true)
    dy_pred, dx_pred = tf.image.image_gradients(y_pred)
    l_edges = tf.reduce_mean(tf.abs(dy_pred - dy_true) + tf.abs(dx_pred - dx_true), axis=-1)
    return 1.0 - l_edges

def loss_depth_wise(y_true, y_pred):
    l_depth = tf.reduce_mean(tf.abs(y_pred - y_true), axis=-1)
    return l_depth

def acc_depth_wise(y_true, y_pred):
    l_depth = tf.reduce_mean(tf.abs(y_pred - y_true), axis=-1)
    return 1.0 - l_depth

def loss_final(y_true, y_pred):
    return alpha1 * ssi_loss(y_true, y_pred) + alpha2 * loss_smooth(y_true, y_pred) + alpha3 * loss_depth_wise(y_true, y_pred)

def acc_final(y_true, y_pred):
    return 1.0 - (alpha1 * ssi_loss(y_true, y_pred) + alpha2 * loss_smooth(y_true, y_pred) + alpha3 * loss_depth_wise(y_true, y_pred))

def convex_comb_loss(alpha1, alpha2, alpha3):
    def loss_final(y_true, y_pred):
        return alpha1 * ssi_loss(y_true, y_pred) + alpha2 * loss_smooth(y_true, y_pred) + alpha3 * loss_depth_wise(y_true, y_pred)
    return loss_final

def convex_comb_acc(alpha1, alpha2, alpha3):
    def acc_final(y_true, y_pred):
        return 1.0 - (alpha1 * ssi_loss(y_true, y_pred) + alpha2 * loss_smooth(y_true, y_pred) + alpha3 * loss_depth_wise(y_true, y_pred))
    return acc_final


def plot_fig(el, num, out_name, colormap='gray' ):
    fig = plt.figure(figsize=(3.84, 1.92), dpi=100, frameon=False)
    ax = fig.add_axes([0, 0, 1, 1])
    ax.axis('off')

    plt.imshow(el[num][0].reshape(IMG_HEIGHT, IMG_WIDTH), cmap=colormap)
    plt.savefig(out_name)

cc_loss =  convex_comb_loss(alpha1, alpha2, alpha3)
cc_acc  =  convex_comb_acc(alpha1, alpha2, alpha3)




autoencoder_R = tf.keras.models.load_model('./run_stasis/stasis_right_net/right_net.hdf5',
                                         custom_objects={'ssi_loss': ssi_loss,
                                                         'ssi_acc': ssi_acc,
                                                         'cc_acc': cc_acc,
                                                         'loss_final':loss_final,
                                                         'acc_final':acc_final})


autoencoder_L = tf.keras.models.load_model('./run_stasis/stasis_left_net/left_net.hdf5',
                                         custom_objects={'ssi_loss': ssi_loss,
                                                         'ssi_acc': ssi_acc,
                                                         'cc_acc': cc_acc,
                                                         'loss_final':loss_final,
                                                         'acc_final':acc_final})


stereo_matching = tf.keras.models.load_model('./run_stasis/disparity_net/030.hdf5',
                                           custom_objects={'ssi_loss': ssi_loss,
                                                           'ssi_acc': ssi_acc,
                                                           'cc_acc': cc_acc,
                                                           'loss_final': loss_final,
                                                           'acc_final': acc_final})

img_l = []
for el in left_imgs:
    img1 = mpimg.imread(dir_left + el)
    img1 = img1[:, :, 0].reshape([1, IMG_HEIGHT, IMG_WIDTH, 1])
    img_l.append(img1)

img_r = []
for el in right_imgs:
    img2 = mpimg.imread(dir_right + el)
    img2 = img2[:, :, 0].reshape([1, IMG_HEIGHT, IMG_WIDTH, 1])
    img_r.append(img2)




dec_L =  []
dec_R =  []
disp_l = []
for i in range(0, len(img_l), 1):
    startTime = time.time()
    decoded_imgs_L = autoencoder_L.predict([img_l[i]])
    decoded_imgs_R = autoencoder_R.predict([img_r[i]])
    matched_depths = stereo_matching.predict([decoded_imgs_L, decoded_imgs_R])
    dec_L.append(decoded_imgs_L)
    dec_R.append(decoded_imgs_R)
    disp_l.append(matched_depths)
    elapsedTime = time.time() - startTime
    print('{} [{}] finished in {} ms'.format("StaSiS-Net: ", left_imgs[i].split(".")[0], int(elapsedTime * 1000)))
    name_l = args.out_dir + left_imgs[i].split(".")[0]  + "_l"
    name_r = args.out_dir + right_imgs[i].split(".")[0] + "_r"
    name_d = args.out_dir + left_imgs[i].split(".")[0]  + "_disp"
    plot_fig(dec_L,  i, name_l, 'inferno')
    plot_fig(dec_R,  i, name_r, 'inferno')
    plot_fig(disp_l, i, name_d, 'inferno')


