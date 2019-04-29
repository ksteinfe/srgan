from easydict import EasyDict as edict
import json

config = edict()
config.TRAIN = edict()

## Adam
config.TRAIN.batch_size = 16 # was originally 16, best to keep this a sqrt
config.TRAIN.lr_init = 1e-4
config.TRAIN.beta1 = 0.9

## initialize G
config.TRAIN.n_epoch_init = 100
    # config.TRAIN.lr_decay_init = 0.1
    # config.TRAIN.decay_every_init = int(config.TRAIN.n_epoch_init / 2)

## adversarial learning (SRGAN)
config.TRAIN.n_epoch = 1000
config.TRAIN.lr_decay = 0.1
config.TRAIN.decay_every = int(config.TRAIN.n_epoch / 2)


## train set location
config.TRAIN.hr_img_path = r'X:\Box Sync\RRSYNC\datasets_derived\190207_srgan_pano\train_hr'
config.TRAIN.lr_img_path = False

config.VALID = edict()
## test set location
config.VALID.hr_img_path = False
config.VALID.lr_img_path = False

'''
## train set location
config.TRAIN.hr_img_path = r'Y:\GitHub\srgan\training_sets\DIV2K\train_hr'
config.TRAIN.lr_img_path = r'Y:\GitHub\srgan\training_sets\DIV2K\train_lr'

config.VALID = edict()
## test set location
config.VALID.hr_img_path = r'Y:\GitHub\srgan\training_sets\DIV2K\valid_hr'
config.VALID.lr_img_path = r'Y:\GitHub\srgan\training_sets\DIV2K\valid_lr'
'''

def log_config(filename, cfg):
    with open(filename, 'w') as f:
        f.write("================================================\n")
        f.write(json.dumps(cfg, indent=4))
        f.write("\n================================================\n")
