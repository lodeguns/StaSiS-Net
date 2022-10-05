from __future__ import absolute_import
from __future__ import print_function
from tensorflow.compat.v1 import ConfigProto
from tensorflow.compat.v1 import InteractiveSession
from os import path

from tensorflow.python.keras.layers import Conv3D, UpSampling3D, MaxPooling3D

config = ConfigProto()
config.gpu_options.allow_growth = True
session = InteractiveSession(config=config)
import os
from os import path
import argparse
import tensorflow as tf
#import tensorflow_addons as tfa

from tensorflow.keras.models import Model, load_model
from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, UpSampling2D, Flatten, Dense, Activation, Concatenate, Dot
from tensorflow.keras.callbacks import TensorBoard, LearningRateScheduler
import math
from datetime import datetime
import pathlib


print("Francesco Bardozzo - 25 Agosto 2020 ")
parser = argparse.ArgumentParser(description='Stereo depth enstimation')
parser.add_argument('--train_setting_flow',        type=int,   help='0 = LtoR (Basic), 1 = RtoL (Basic), 2=LRtoR (Twin), 3=RLtoL (Twin)', default=0)
parser.add_argument('--simple_model',              type=int,   help='0 = base_network_model, 1 = simple network model', default=0)
parser.add_argument('--shared_model',              type=int,   help='(Twin) Shared_model = 0, False=1, Transfer Single Branches Learning = 2', default=0)
parser.add_argument('--exp_name',                  type=str,   help='name of the experiment', default="Experiment0Marconi100")
parser.add_argument('--input_height',              type=int,   help='input height', default=192)
parser.add_argument('--input_width',               type=int,   help='input width', default=384)
parser.add_argument('--batch_size',                type=int,   help='batch size', default=30)
parser.add_argument('--num_epochs',                type=int,   help='number of epochs', default=100)
parser.add_argument('--lr',                        type=float, help='initial learning rate', default=0.01)
parser.add_argument('--train_size_set',            type=int,   help='train_size_set', default=34240)
parser.add_argument('--test_size_set',             type=int,   help='test_size_set', default=7191)
parser.add_argument('--k1',                        type=int,   help='k_base def 32', default=32)
parser.add_argument('--d1',                        type=int,   help='dense neurons base 300', default=300)
parser.add_argument('--alpha1',                    type=float, help='alpha1', default=0.2)
parser.add_argument('--alpha2',                    type=float, help='alpha2', default=0.7)
parser.add_argument('--alpha3',                    type=float, help='alpha3', default=0.1)
parser.add_argument('--learning_rate_decay',       type=int,   help='LR decay yes (1), no(0)', default=0)
parser.add_argument('--learning_rate_plateau',     type=int,   help='LR reduce LR plateau monitoring loss yes (1), no(0)', default=0)
parser.add_argument('--decay_factor_lr',           type=float, help='decay learning rate', default=0.9)
parser.add_argument('--patience_decay',            type=int,   help='patience decay learning rate', default=5)
parser.add_argument('--selected_imgs',             type=int,   help='Img depths (1) or the whole dataset (0)', default=0)
parser.add_argument('--continue_fit',              type=str,   help="Continue training no(0), yes the name of the checkpoint ex 030", default='0')
parser.add_argument('--gpus',                      type=int,   default=1, help='The number of GPUs to use')
parser.add_argument('--gpuids',                    type=str,   default='0', help='IDs of GPUs to use 0,1,2,3,4,5,6,7')

args = parser.parse_args()

# Inform about multi-gpu training
if args.gpus == 1:
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpuids
    print('-------  Will use GPU ' + args.gpuids) #usage 0,1,2,3,4,5,6,7
else:
    print('-------- Will use ' + str(args.gpus) + ' GPUs.')
# Create a MirroredStrategy.
strategy = tf.distribute.MirroredStrategy()
print('--- Number of devices: {}'.format(strategy.num_replicas_in_sync))



EPOCHS     = args.num_epochs
BATCH_SIZE = args.batch_size
IMG_HEIGHT = args.input_height
IMG_WIDTH  = args.input_width
LEARNING_RATE = args.lr
LEFT_RIGHT_TRAIN = args.train_size_set
LEFT_RIGHT_TEST  = args.test_size_set
margin  =  1 #contrastive loss
alpha1  =  args.alpha1
alpha2  =  args.alpha2
alpha3  =  args.alpha3
decay_factor = args.decay_factor_lr
patience_lr  = args.patience_decay
k1  = args.k1
d1  = args.k1
shared_model = args.shared_model
train_setting_flow = args.train_setting_flow
learning_rate_decay = args.learning_rate_decay
learning_rate_plateau = args.learning_rate_plateau
selected_imgs = args.selected_imgs
continue_fit  = args.continue_fit
simple_model = args.simple_model

def all_subdirs_of(b='.'):
  result = []
  for d in os.listdir(b):
    bd = os.path.join(b, d)
    if os.path.isdir(bd): result.append(bd)
  return result


def tensor_checkpoints_and_retrain(note, continue_fit = 0):
    if continue_fit != '0':
        all_subdirs = all_subdirs_of('checkpoints')
        latest_subdir = max(all_subdirs, key=os.path.getmtime).split('/')
        name_folder = latest_subdir[1] + "_continue_fit"
    else:
        name_folder = str(datetime.now().strftime("D%Y%m%d%H%M%S")) + "E" \
                      + str(EPOCHS) + "BS" + str(BATCH_SIZE) + "LR" + str(LEARNING_RATE).replace('.', '') + note

    os.makedirs('tensorboard/' + str(name_folder), exist_ok=True)
    os.makedirs('saved_models/' + str(name_folder), exist_ok=True)
    os.makedirs('checkpoints/' + str(name_folder), exist_ok=True)

    # TENSORBOARD
    # tensorboard --logdir=/tmp/depth_trace

    log_dir = 'tensorboard/' + name_folder + '/'
    tb_callback = TensorBoard(log_dir=log_dir, histogram_freq=1)

    # CHECKPOINT
    checkpoint_path = "checkpoints/" + str(name_folder) + "/{epoch:03d}.hdf5"
    checkpoint_dir = path.dirname(checkpoint_path)
    model_path = "saved_models/" + str(name_folder)

    cp_callback = tf.keras.callbacks.ModelCheckpoint(
        filepath=checkpoint_path,
        monitor='ssi_acc',
        verbose=1,
        save_best_only=False,
        mode='max',
        save_freq='epoch')



    return name_folder, log_dir, tb_callback, checkpoint_path, checkpoint_dir, cp_callback, model_path


def load_LR_net(path_hdf5):
    print("Loading LR network model")
    LR_mod = tf.keras.models.load_model(path_hdf5,
                                        custom_objects={'ssi_loss': ssi_loss,
                                                        'ssi_acc': ssi_acc,
                                                        'cc_acc': cc_acc,
                                                        'loss_final': loss_final,
                                                        'acc_final': acc_final})

    LR_mod = Model(LR_mod.input, LR_mod.layers[-1].output)
    print(LR_mod.summary())

    return LR_mod


def load_RL_net(path_hdf5):
    print("Loading RL network model")
    RL_mod = tf.keras.models.load_model(path_hdf5,
                                        custom_objects={'ssi_loss': ssi_loss,
                                                        'ssi_acc': ssi_acc,
                                                        'cc_acc': cc_acc,
                                                        'loss_final': loss_final,
                                                        'acc_final': acc_final})

    RL_mod = Model(RL_mod.input, RL_mod.layers[-1].output)
    print(RL_mod.summary())
    return RL_mod


def load_model_from_checkpoint(path_hdf5):
    #example path hdf5 './checkpoints/D20200525221639E50BS60LR0001Skip-Sub2/046.hdf5'
    siamese = tf.keras.models.load_model(path_hdf5,
                                         custom_objects={'ssi_loss': ssi_loss,
                                                         'ssi_acc': ssi_acc,
                                                         'cc_acc': cc_acc,
                                                         'loss_final':loss_final,
                                                         'acc_final':acc_final})
    return siamese



def load_path_dataset(selected_imgs):
    if selected_imgs == 0:
        ''' Train set Left / Right'''
        L1 = pathlib.Path('data/train_set/L')
        R1 = pathlib.Path('data/train_set/R')

        ''' Test set Left / Right'''
        L2 = pathlib.Path('data/test_set/L')
        R2 = pathlib.Path('data/test_set/R')
    else:
        ''' Train set Depth enst Left / Right'''
        L1 = pathlib.Path('data_depth/train_set_depth/L')
        R1 = pathlib.Path('data_depth/train_set_depth/R')

        ''' Test set Depth enst  Left / Right'''
        L2 = pathlib.Path('data_depth/test_set_depth/L')
        R2 = pathlib.Path('data_depth/test_set_depth/R')
    return L1, R1, L2, R2




# The 1./255 is to convert from uint8 to float32 in range [0,1]
L1, R1, L2, R2 = load_path_dataset(selected_imgs)
datagen  = tf.keras.preprocessing.image.ImageDataGenerator(rescale=1.0 / 255.0)

dt0 = datagen.flow_from_directory(L1, class_mode=None,
                                  batch_size=BATCH_SIZE,
                                  color_mode="grayscale",
                                  shuffle=True,
                                  seed=60386,
                                  target_size=(IMG_HEIGHT, IMG_WIDTH))

dt1 = datagen.flow_from_directory(R1, class_mode=None,
                                  batch_size=BATCH_SIZE,
                                  color_mode="grayscale",
                                  shuffle=True,
                                  seed=60386,
                                  target_size=(IMG_HEIGHT, IMG_WIDTH))

dtt0 = datagen.flow_from_directory(L2, class_mode=None,
                                       batch_size=BATCH_SIZE,
                                       color_mode="grayscale",
                                       shuffle=True,
                                       seed=60386,
                                       target_size=(IMG_HEIGHT, IMG_WIDTH))

dtt1 = datagen.flow_from_directory(R2, class_mode=None,
                                       batch_size=BATCH_SIZE,
                                       color_mode="grayscale",
                                       shuffle=True,
                                       seed=60386,
                                       target_size=(IMG_HEIGHT, IMG_WIDTH))





'''
def combine_gen(dt0, dt1, train_setting_flow):
    while True: #LR
        a = next(dt0)
        b = next(dt1)
        yield a, b
'''

def create_base_network(input_shape, k1=32):
    '''Base network to be shared (eq. to feature extraction).
    '''
    input = Input(shape=input_shape)

    encoded_t = Conv2D(k1, (3, 3), activation='elu', padding='same')(input)
    encoded = MaxPooling2D((2, 2), padding='same')(encoded_t)

    encoded_t = Conv2D(k1, (3, 3), activation='elu', padding='same')(encoded)
    encoded = MaxPooling2D((2, 2), padding='same')(encoded_t)

    encoded_s = Conv2D(k1, (3, 3), activation='elu', padding='same')(encoded)
    encoded = MaxPooling2D((2, 2), padding='same')(encoded_s)

    encoded_skip = Conv2D(k1*2, (3, 3), activation='elu', padding='same')(encoded)
    encoded      = Conv2D(k1*4, (3, 3), activation='elu', padding='same')(encoded_skip)
    decoded      = Conv2D(k1*8, (3, 3), activation='elu', padding='same')(encoded)  # <- questa potrebbe essere relu+
    decoded      = Conv2D(k1*4, (3, 3), activation='elu', padding='same')(decoded)
    decoded      = Conv2D(k1*2, (3, 3), activation='elu', padding='same')(decoded)
    #decoded      = tf.keras.layers.add([decoded, encoded_skip])

    decoded = UpSampling2D((2, 2), interpolation='nearest')(decoded)
    decoded = Conv2D(k1, (3, 3), activation='elu', padding='same')(decoded)
    decoded      = tf.keras.layers.add([decoded, encoded_s])

    decoded = UpSampling2D((2, 2), interpolation='nearest')(decoded)
    decoded = Conv2D(k1, (3, 3), activation='elu', padding='same')(decoded)
    decoded      = tf.keras.layers.add([decoded, encoded_t])

    decoded = UpSampling2D((2, 2), interpolation='nearest')(decoded)
    decoded = Conv2D(k1, (3, 3), activation='elu', padding='same')(decoded)
    #decoded      = tf.keras.layers.add([decoded, encoded_t])

    decoded = Conv2D(1, (3, 3), activation='sigmoid', padding='same')(decoded)
    return Model(input, decoded)


def create_base_network_v2(input_shape, k1=32):
    '''Base network to be shared (eq. to feature extraction).
    '''
    input = Input(shape=input_shape)

    #enc1
    encoded   = Conv2D(k1, (3, 3), activation='elu', padding='same')(input)
    encoded_1 = Conv2D(k1, (3, 3), activation='elu', padding='same')(encoded)
    encoded   = MaxPooling2D((2, 2), padding='same')(encoded_1)

    encoded   = Conv2D(k1, (3, 3), activation='elu', padding='same')(encoded)
    encoded_2 = Conv2D(k1, (3, 3), activation='elu', padding='same')(encoded)
    encoded   = MaxPooling2D((2, 2), padding='same')(encoded_2)

    encoded   = Conv2D(k1, (3, 3), activation='elu', padding='same')(encoded)
    encoded   = Conv2D(k1, (3, 3), activation='elu', padding='same')(encoded)
    encoded_3 = Conv2D(k1, (3, 3), activation='elu', padding='same')(encoded)
    encoded   = MaxPooling2D((2, 2), padding='same')(encoded_3)

    encoded = Conv2D(k1*2, (3, 3), activation='elu', padding='same')(encoded)
    encoded = Conv2D(k1*4, (3, 3), activation='elu', padding='same')(encoded)
    decoded = Conv2D(k1*8, (3, 3), activation='elu', padding='same')(encoded)  # <- questa potrebbe essere relu+
    decoded = Conv2D(k1*4, (3, 3), activation='elu', padding='same')(decoded)
    decoded = Conv2D(k1*2, (3, 3), activation='elu', padding='same')(decoded)


    decoded = UpSampling2D((2, 2), interpolation='nearest')(decoded)
    decoded = Conv2D(k1, (3, 3), activation='elu', padding='same')(decoded)
    decoded = Conv2D(k1, (3, 3), activation='elu', padding='same')(decoded)
    decoded = Conv2D(k1, (3, 3), activation='elu', padding='same')(decoded)
    decoded = tf.keras.layers.add([decoded, encoded_3])

    decoded = UpSampling2D((2, 2), interpolation='nearest')(decoded)
    decoded = Conv2D(k1, (3, 3), activation='elu', padding='same')(decoded)
    decoded = Conv2D(k1, (3, 3), activation='elu', padding='same')(decoded)
    decoded = tf.keras.layers.add([decoded, encoded_2])

    decoded = UpSampling2D((2, 2), interpolation='nearest')(decoded)
    decoded = Conv2D(k1, (3, 3), activation='elu', padding='same')(decoded)
    decoded = Conv2D(k1, (3, 3), activation='elu', padding='same')(decoded)
    decoded      = tf.keras.layers.add([decoded, encoded_1])

    decoded = Conv2D(1, (3, 3), activation='sigmoid', padding='same')(decoded)

    return Model(input, decoded)


def create_base_network_v3(input_shape, k1=32):
    '''Base network to be shared (eq. to feature extraction).
    '''
    input = Input(shape=input_shape)

    #enc1
    encoded   = Conv2D(k1, (3, 3), activation='elu', padding='same')(input)
    encoded   = Conv2D(k1, (3, 3), activation='elu', padding='same')(encoded)
    encoded   = Conv2D(k1, (3, 3), activation='elu', padding='same')(encoded)
    encoded   = MaxPooling2D((2, 2), padding='same')(encoded)

    encoded   = Conv2D(k1, (3, 3), activation='elu', padding='same')(encoded)
    encoded   = Conv2D(k1, (3, 3), activation='elu', padding='same')(encoded)
    encoded   = Conv2D(k1, (3, 3), activation='elu', padding='same')(encoded)
    encoded   = MaxPooling2D((2, 2), padding='same')(encoded)

    encoded   = Conv2D(k1, (3, 3), activation='elu', padding='same')(encoded)
    encoded   = Conv2D(k1, (3, 3), activation='elu', padding='same')(encoded)
    encoded_2 = Conv2D(k1, (3, 3), activation='elu', padding='same')(encoded)
    encoded   = MaxPooling2D((2, 2), padding='same')(encoded_2)

    encoded   = Conv2D(k1, (3, 3), activation='elu', padding='same')(encoded)
    encoded   = Conv2D(k1, (3, 3), activation='elu', padding='same')(encoded)
    encoded_3 = Conv2D(k1, (3, 3), activation='elu', padding='same')(encoded)
    encoded   = MaxPooling2D((2, 2), padding='same')(encoded_3)

    encoded = Conv2D(k1*2, (3, 3), activation='elu', padding='same')(encoded)
    encoded = Conv2D(k1*4, (3, 3), activation='elu', padding='same')(encoded)
    decoded = Conv2D(k1*8, (3, 3), activation='elu', padding='same')(encoded)  # <- questa potrebbe essere relu+
    decoded = Conv2D(k1*4, (3, 3), activation='elu', padding='same')(decoded)
    decoded = Conv2D(k1*2, (3, 3), activation='elu', padding='same')(decoded)


    decoded = UpSampling2D((2, 2), interpolation='nearest')(decoded)
    decoded = Conv2D(k1, (3, 3), activation='elu', padding='same')(decoded)
    decoded = Conv2D(k1, (3, 3), activation='elu', padding='same')(decoded)
    decoded = Conv2D(k1, (3, 3), activation='elu', padding='same')(decoded)
    decoded = tf.keras.layers.add([decoded, encoded_3])

    decoded = UpSampling2D((2, 2), interpolation='nearest')(decoded)
    decoded = Conv2D(k1, (3, 3), activation='elu', padding='same')(decoded)
    decoded = Conv2D(k1, (3, 3), activation='elu', padding='same')(decoded)
    decoded = Conv2D(k1, (3, 3), activation='elu', padding='same')(decoded)
    decoded = tf.keras.layers.add([decoded, encoded_2])

    decoded = UpSampling2D((2, 2), interpolation='nearest')(decoded)
    decoded = Conv2D(k1, (3, 3), activation='elu', padding='same')(decoded)
    decoded = Conv2D(k1, (3, 3), activation='elu', padding='same')(decoded)
    decoded = Conv2D(k1, (3, 3), activation='elu', padding='same')(decoded)

    decoded = UpSampling2D((2, 2), interpolation='nearest')(decoded)
    decoded = Conv2D(k1, (3, 3), activation='elu', padding='same')(decoded)
    decoded = Conv2D(k1, (3, 3), activation='elu', padding='same')(decoded)
    decoded = Conv2D(k1, (3, 3), activation='elu', padding='same')(decoded)

    decoded = Conv2D(1, (3, 3), activation='sigmoid', padding='same')(decoded)

    return Model(input, decoded)


def create_base_network2(input_shape, k1=32):
    '''Base network to be shared (eq. to feature extraction).
    '''
    input = Input(shape=input_shape)

    encoded_t = Conv2D(k1, (3, 3), activation='elu', padding='same')(input)
    encoded = MaxPooling2D((2, 2), padding='same')(encoded_t)

    encoded_t = Conv2D(k1, (3, 3), activation='elu', padding='same')(encoded)
    encoded = MaxPooling2D((2, 2), padding='same')(encoded_t)

    encoded_s = Conv2D(k1, (3, 3), activation='elu', padding='same')(encoded)
    encoded = MaxPooling2D((2, 2), padding='same')(encoded_s)

    encoded_skip = Conv2D(k1 * 2, (3, 3), activation='elu', padding='same')(encoded)
    encoded = Conv2D(k1 * 4, (3, 3), activation='elu', padding='same')(encoded_skip)
    decoded = Conv2D(k1 * 8, (3, 3), activation='elu', padding='same')(encoded)  # <- questa potrebbe essere relu+
    decoded = Conv2D(k1 * 4, (3, 3), activation='elu', padding='same')(decoded)
    decoded = Conv2D(k1 * 2, (3, 3), activation='elu', padding='same')(decoded)
    # decoded      = tf.keras.layers.add([decoded, encoded_skip])

    decoded = UpSampling2D((2, 2), interpolation='nearest')(decoded)
    decoded = Conv2D(k1, (3, 3), activation='elu', padding='same')(decoded)
    decoded = tf.keras.layers.add([decoded, encoded_s])

    decoded = UpSampling2D((2, 2), interpolation='nearest')(decoded)
    decoded = Conv2D(k1, (3, 3), activation='elu', padding='same')(decoded)
    decoded = tf.keras.layers.add([decoded, encoded_t])

    decoded = UpSampling2D((2, 2), interpolation='nearest')(decoded)
    decoded = Conv2D(k1, (3, 3), activation='elu', padding='same')(decoded)
    # decoded      = tf.keras.layers.add([decoded, encoded_t])

    decoded = Conv2D(1, (3, 3), activation='sigmoid', padding='same')(decoded)
    return Model(input, decoded)





def create_res_network(input_shape, k1=32):
    '''Base/ResNet101 network'''
    input = Input(shape=input_shape)
    x = tf.keras.applications.resnet.preprocess_input(input)
    core =  tf.keras.applications.ResNet101(include_top=False, weights='imagenet', pooling=None)
    x = core(x)

    encoded = Conv2D(1, (3, 3), activation='elu', padding='same')(x)
    encoded = Conv2D(1, (3, 3), activation='sigmoid', padding='same')(encoded)
    decoded = Conv2D(1, (3, 3), activation='sigmoid', padding='same')(encoded)
    mod = Model(input, decoded)
    print(mod.summary())
    return mod

def simple_network(input_shape, k1=32):

    input_img = Input(shape=input_shape)

    encoded = Conv2D(32, (3, 3), activation='elu', padding='same')(input_img)
    encoded = MaxPooling2D((2, 2), padding='same')(encoded)

    encoded = Conv2D(32, (3, 3), activation='elu', padding='same')(encoded)
    encoded_skip = MaxPooling2D((2, 2), padding='same')(encoded)

    encoded = Conv2D(32, (3, 3), activation='elu', padding='same')(encoded_skip)

    encoded = Conv2D(64, (3, 3), activation='elu', padding='same')(encoded)
    encoded = Conv2D(128, (3, 3), activation='elu', padding='same')(encoded)
    decoded = Conv2D(256, (3, 3), activation='elu', padding='same')(encoded)  # <- questa potrebbe essere relu+
    decoded = Conv2D(128, (3, 3), activation='elu', padding='same')(decoded)
    decoded = Conv2D(64, (3, 3), activation='elu', padding='same')(decoded)

    decoded = Conv2D(32, (3, 3), activation='elu', padding='same')(decoded)  # additional layer
    decoded = tf.keras.layers.add([decoded, encoded_skip])

    decoded = UpSampling2D((2, 2), interpolation='nearest')(decoded)
    decoded = Conv2D(32, (3, 3), activation='elu', padding='same')(decoded)

    decoded = UpSampling2D((2, 2), interpolation='nearest')(decoded)
    decoded = Conv2D(32, (3, 3), activation='elu', padding='same')(decoded)

    decoded = Conv2D(1, (3, 3), activation='sigmoid', padding='same')(decoded)
    return Model(input_img, decoded)


def hamy_network(input_shape, k1=32):
    input = Input(shape=input_shape)
    #conv1
    encoded  = Conv2D(64, (3, 3), activation='elu', padding='same')(input)
    encoded  = Conv2D(64, (3, 3), activation='elu', padding='same')(encoded)
    encoded  = MaxPooling2D((2, 2), padding='same')(encoded)
    #conv2
    encoded  = Conv2D(128, (3, 3), activation='elu', padding='same')(encoded)
    encoded  = Conv2D(128, (3, 3), activation='elu', padding='same')(encoded)
    encoded  = MaxPooling2D((2, 2), padding='same')(encoded)
    #conv3
    encoded  = Conv2D(256, (3, 3), activation='elu', padding='same')(encoded)
    encoded  = Conv2D(256, (3, 3), activation='elu', padding='same')(encoded)
    encoded  = Conv2D(256, (3, 3), activation='elu', padding='same')(encoded)
    encoded  = MaxPooling2D((2, 2), padding='same')(encoded)
    #conv4
    encoded  = Conv2D(512, (3, 3), activation='elu', padding='same')(encoded)
    encoded  = Conv2D(512, (3, 3), activation='elu', padding='same')(encoded)
    encoded  = Conv2D(512, (3, 3), activation='elu', padding='same')(encoded)
    encoded  = MaxPooling2D((2, 2), padding='same')(encoded)

    #conv5
    encoded  = Conv2D(512, (3, 3), activation='elu', padding='same')(encoded)
    encoded  = Conv2D(512, (3, 3), activation='elu', padding='same')(encoded)
    encoded  = Conv2D(512, (3, 3), activation='elu', padding='same')(encoded)
    encoded  = MaxPooling2D((2, 2), padding='same')(encoded)

    decoded = Conv2D(512, (3, 3), activation='elu', padding='same')(encoded)

    decoded = UpSampling2D((2, 2), interpolation='nearest')(decoded)
    decoded  = Conv2D(512, (3, 3), activation='elu', padding='same')(decoded)
    decoded  = Conv2D(512, (3, 3), activation='elu', padding='same')(decoded)
    decoded  = Conv2D(512, (3, 3), activation='elu', padding='same')(decoded)

    decoded = UpSampling2D((2, 2), interpolation='nearest')(decoded)
    decoded  = Conv2D(512, (3, 3), activation='elu', padding='same')(decoded)
    decoded  = Conv2D(512, (3, 3), activation='elu', padding='same')(decoded)
    decoded  = Conv2D(512, (3, 3), activation='elu', padding='same')(decoded)

    decoded = UpSampling2D((2, 2), interpolation='nearest')(decoded)
    decoded  = Conv2D(256, (3, 3), activation='elu', padding='same')(decoded)
    decoded  = Conv2D(256, (3, 3), activation='elu', padding='same')(decoded)
    decoded  = Conv2D(256, (3, 3), activation='elu', padding='same')(decoded)

    decoded = UpSampling2D((2, 2), interpolation='nearest')(decoded)
    decoded  = Conv2D(128, (3, 3), activation='elu', padding='same')(decoded)
    decoded  = Conv2D(128, (3, 3), activation='elu', padding='same')(decoded)

    decoded = UpSampling2D((2, 2), interpolation='nearest')(decoded)
    decoded  = Conv2D(64, (3, 3), activation='elu', padding='same')(decoded)
    decoded  = Conv2D(64, (3, 3), activation='elu', padding='same')(decoded)

    decoded = Conv2D(1, (3, 3), activation='sigmoid', padding='same')(decoded)
    return Model(input, decoded)




## SSIM
def ssi_loss(y_true, y_pred):
 max_max = tf.reduce_max(tf.stack([tf.reduce_max(y_true), tf.reduce_max(y_pred)], 0))
 return   1.0 - tf.reduce_mean(tf.image.ssim(y_true, y_pred, max_val=max_max, filter_size=3, filter_sigma=1.5, k1=0.01, k2=0.03))

def ssi_acc(y_true, y_pred):
 max_max = tf.reduce_max(tf.stack([tf.reduce_max(y_true), tf.reduce_max(y_pred)], 0))
 return  tf.reduce_mean(tf.image.ssim(y_true, y_pred, max_val=max_max, filter_size=3, filter_sigma=1.5, k1=0.01, k2=0.03))

## SSIM Multiscale
def ssi_loss_ms(y_true, y_pred):
 max_max = tf.reduce_max(tf.stack([tf.reduce_max(y_true), tf.reduce_max(y_pred)], 0))
 return 1.0 - tf.reduce_mean(tf.image.ssim_multiscale(y_true, y_pred, max_val=max_max, filter_size=3, power_factors=[(0.0448, 0.2856, 0.3001, 0.2363, 0.1333)] ))

def ssi_acc_ms(y_true, y_pred):
 max_max = tf.reduce_max(tf.stack([tf.reduce_max(y_true), tf.reduce_max(y_pred)], 0))
 return tf.reduce_mean(tf.image.ssim_multiscale(y_true, y_pred, max_val=max_max, filter_size=3, power_factors=[(0.0448, 0.2856, 0.3001, 0.2363, 0.1333)] ))


#Contrastive Loss
#def acc_contrastive(y_true, y_pred):
#   con_loss =tf.reduce_mean(tfa.losses.contrastive_loss(y_true, y_pred, 1.0))
#   return  1.0 - con_loss


#def loss_contrastive(y_true, y_pred):
#    con_loss = tf.reduce_mean(tfa.losses.contrastive_loss(y_true, y_pred, 1.0))
#    return con_loss

#Smooth loss
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

#Depth-wise loss #Also colled L1 consistency loss
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





#Convex combination of the losses above
def convex_comb_loss(alpha1, alpha2, alpha3):
    def loss_final(y_true, y_pred):
        return alpha1 * ssi_loss(y_true, y_pred) + alpha2 * loss_smooth(y_true, y_pred) + alpha3 * loss_depth_wise(y_true, y_pred)
    return loss_final

def convex_comb_acc(alpha1, alpha2, alpha3):
    def acc_final(y_true, y_pred):
        return 1.0 - (alpha1 * ssi_loss(y_true, y_pred) + alpha2 * loss_smooth(y_true, y_pred) + alpha3 * loss_depth_wise(y_true, y_pred))
    return acc_final






def step_decay(epoch):
   #hamlyin center settings
   initial_lrate = 0.0001
   drop = 0.5
   epochs_drop = 5.0
   lrate = initial_lrate * math.pow(drop, math.floor((1+epoch)/epochs_drop))
   return lrate







def combine_gen(dt0, dt1,train_setting_flow):
    if train_setting_flow == 0:
        while True: #LR
            a = next(dt0)
            b = next(dt1)
            yield a, b
    elif train_setting_flow == 1:
        while True: #RL
            a = next(dt0)
            b = next(dt1)
            yield b, a
    elif train_setting_flow == 2:
        while True: #LR - L
            a = next(dt0)
            b = next(dt1)
            yield (a,b), (b,a)
    elif train_setting_flow == 3:
        while True: #RL - R
            a = next(dt0)
            b = next(dt1)
            yield (b, a), (a,b)





stereo_pair_train = combine_gen( dt0,  dt1, train_setting_flow)
stereo_pair_test  = combine_gen(dtt0, dtt1, train_setting_flow)

#input
input_shape = (IMG_HEIGHT, IMG_WIDTH, 1)

#define personalized loss
cc_loss =  convex_comb_loss(alpha1, alpha2, alpha3)
cc_acc  =  convex_comb_acc(alpha1, alpha2, alpha3)


if learning_rate_decay == 1:
    print("--- Model running with learning rate decay ---")
    decay_rate = LEARNING_RATE / EPOCHS*2
    opt = tf.keras.optimizers.Adadelta(learning_rate=LEARNING_RATE, rho=0.95, epsilon=1e-07, decay= decay_rate, name='adadelta')
else:
    opt = tf.keras.optimizers.Adadelta(learning_rate=LEARNING_RATE, rho=0.95, epsilon=1e-07, name='adadelta')

name_folder, _, tb_callback, checkpoint_path, _, cp_callback, model_path = tensor_checkpoints_and_retrain(args.exp_name, continue_fit)

if learning_rate_plateau ==1:
    print("--- Model running with reduce learning on plateau ---")
    reduce_lr = tf.keras.callbacks.ReduceLROnPlateau(monitor='cc_loss', verbose=1, factor=decay_factor, mode='min', patience=patience_lr, min_lr=0.000000000000000000001)




# Open a strategy scope.
with strategy.scope():
  if train_setting_flow == 2 or train_setting_flow==3:
      print("-- Siamese running --")
      input_a = Input(shape=input_shape)
      input_b = Input(shape=input_shape)

      if shared_model == 0:
          print("--- Siamese Weights Shared Model ---")
          base_network = create_base_network(input_shape, k1)
          print(base_network.summary())
          processed_a = base_network(input_a)
          processed_b = base_network(input_b)
      elif shared_model == 11:
          print("--- Siamese Weights Shared Model v2 ---")
          base_network = create_base_network_v2(input_shape, k1)
          processed_a = base_network(input_a)
          processed_b = base_network(input_b)
      elif shared_model == 12:
          print("--- Siamese Weights Shared Model v3 -- Less Skip ---")
          base_network = create_base_network_v3(input_shape, k1)
          processed_a = base_network(input_a)
          processed_b = base_network(input_b)
      elif shared_model == 1:
          print("--- Siamese Weights Unshared Model ---")
          base_network_left = create_base_network(input_shape, k1)
          base_network_right = create_base_network2(input_shape, k1)
          processed_a = base_network_left(input_a)
          processed_b = base_network_right(input_b)
      else:
          print("--- Siamese with already Learned Branches ---")
          LR_network = load_LR_net('pretrain/LR/139.hdf5')
          RL_network = load_RL_net('pretrain/RL/139.hdf5')
          input_a = Input(shape=input_shape)
          input_b = Input(shape=input_shape)
          processed_a = LR_network(input_a, training=False)
          processed_b = RL_network(input_b, training=False)
          print(processed_a.shape)
          print(processed_b.shape)
          #processed_a = Conv2D(1, (3, 3), activation='sigmoid', padding='same')(x_LR)
          #processed_b = Conv2D(1, (3, 3), activation='sigmoid', padding='same')(x_RL)

      concat_proc = Concatenate(axis=-1)([processed_a, processed_b])
      concat_proc = Dense(d1, activation='elu', kernel_regularizer=tf.keras.regularizers.l1(0.01), bias_regularizer=tf.keras.regularizers.l1(0.01))(concat_proc)
      concat_proc = Dense(d1, activation='elu', kernel_regularizer=tf.keras.regularizers.l1(0.01), bias_regularizer=tf.keras.regularizers.l1(0.01))(concat_proc)
      concat_proc = Dense(d1, activation='elu', kernel_regularizer=tf.keras.regularizers.l1(0.01), bias_regularizer=tf.keras.regularizers.l1(0.01))(concat_proc)
      concat_proc = Conv2D(1, (3, 3), activation='sigmoid', padding='same')(concat_proc)
      autoencoder = Model([input_a, input_b], [concat_proc])
      print(autoencoder.summary())

  elif train_setting_flow == 0 or train_setting_flow ==1:
      print("--- Single Branch Model ---")
      if simple_model == 0:
        print("--- Model with 2 Skip conn ---")
        autoencoder = create_base_network(input_shape, k1)
      if simple_model == 2:
        print("--- Hamlyn Center ---")
        autoencoder = hamy_network(input_shape, k1)
      if simple_model ==3:
        print("--- umm")
        autoencoder = create_res_network(input_shape, k1)
        print(autoencoder.summary())
      else:
        print("--- Model with 1 Skip conn ---")
        autoencoder = simple_network(input_shape, 32) #less accurate

  autoencoder.compile(optimizer=opt, loss=cc_loss, metrics=[ssi_acc, cc_acc, 'mse'])

#print(autoencoder.summary())



autoencoder.save_weights(checkpoint_path.format(epoch=0, ssi_acc=0))



if continue_fit != '0':
    print("enter in continue fit...")
    all_subdirs = all_subdirs_of('checkpoints')
    latest_subdir = max(all_subdirs, key=os.path.getmtime).split('/')
    name_folder = latest_subdir  + "/" +  continue_fit  + ".hdf5"
    autoencoder = tf.keras.models.load_model(name_folder,
                                             custom_objects={'cc_loss': cc_loss,
                                                             'ssi_acc': ssi_acc,
                                                             'cc_acc' : cc_acc})

    autoencoder.fit(stereo_pair_train,
                steps_per_epoch=LEFT_RIGHT_TRAIN // BATCH_SIZE,
                epochs=EPOCHS,
                shuffle=True,
                validation_data=stereo_pair_test,
                validation_steps=LEFT_RIGHT_TEST // BATCH_SIZE,
                callbacks=[ cp_callback, tb_callback],
                verbose=1 )
else:
    autoencoder.fit(stereo_pair_train,
                steps_per_epoch=LEFT_RIGHT_TRAIN//BATCH_SIZE,
                epochs=EPOCHS,
                shuffle=True,
                validation_data= stereo_pair_test,
                validation_steps=LEFT_RIGHT_TEST//BATCH_SIZE,
                callbacks = [cp_callback, tb_callback],
                verbose=1 )

    '''lrate = LearningRateScheduler(step_decay)  #this is lrate with step decay
    autoencoder.fit(stereo_pair_train,
                steps_per_epoch=LEFT_RIGHT_TRAIN//BATCH_SIZE,
                epochs=EPOCHS,
                shuffle=True,
                validation_data= stereo_pair_test,
                validation_steps=LEFT_RIGHT_TEST//BATCH_SIZE,
                callbacks = [lrate, cp_callback, tb_callback],
                verbose=1 )'''


#autoencoder.save(model_path)
print(name_folder)
