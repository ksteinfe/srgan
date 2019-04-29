#! /usr/bin/python
# -*- coding: utf8 -*-
# KSTEINFE adaptation for enlarging images given a trained model.
import os, time, argparse


MAX_SIZE = 800

def upsample_images(pth_src, pth_dst, pth_checkpoint):
    import numpy as np
    import scipy

    import tensorlayer as tl
    import tensorflow as tf
    from model import SRGAN_g

    print("==== UPSAMPLING IMAGES")
    ## create folders to save result images
    tl.files.exists_or_mkdir(pth_dst)


    t_image = tf.placeholder('float32', [1, None, None, 3], name='input_image')
    net_g = SRGAN_g(t_image, is_train=False, reuse=False)

    # Restore Generator
    sess = tf.Session(config=tf.ConfigProto(allow_soft_placement=True, log_device_placement=False))
    tl.layers.initialize_global_variables(sess)
    tl.files.load_and_assign_npz(sess=sess, name=pth_checkpoint, network=net_g)
    print("loaded srgan model from {}".format(pth_checkpoint))


    valid_lr_img_list = sorted(tl.files.load_file_list(path=pth_src, regx='.*.(jpg|png)', printable=False))
    #valid_lr_imgs = tl.vis.read_images(valid_lr_img_list, path=pth_src, n_threads=32)
    print("found {} valid images to upscale...".format(len(valid_lr_img_list)))
    for n,fname in enumerate(valid_lr_img_list):
        bname = os.path.splitext(os.path.join(pth_src,fname))[0]
        img_src = scipy.misc.imread(os.path.join(pth_src,fname), mode='RGB')
        img_src = (img_src / 127.5) - 1  # rescale to ［－1, 1]
        size = img_src.shape
        if size[0] > MAX_SIZE or size[1] > MAX_SIZE:
            print("Image is too big ({}x{}). Skipping.".format(size[0],size[1]))
            continue

        # Evaluate
        start_time = time.time()
        img_dst = sess.run(net_g.outputs, {t_image: [img_src]})
        img_dst = ((img_dst + 1)/2.0)*255 # rescale to [0,255]
        img_dst = img_dst.astype(np.uint8) # convert to unsigned int for saving to image
        print("{} of {}\tUpsampling {} from {}x{} to {}x{} took {:.2f}s".format(n,len(valid_lr_img_list),fname,size[0],size[1],img_dst.shape[1],img_dst.shape[2],time.time() - start_time))
        tl.vis.save_image(img_dst[0], os.path.join(pth_dst,fname))


if __name__ == '__main__':
    """Checks if a path is an actual directory"""
    def is_dir(pth):
        if not os.path.isdir(pth):
            msg = "{0} is not a directory".format(pth)
            raise argparse.ArgumentTypeError(msg)
        else:
            return os.path.abspath(os.path.realpath(os.path.expanduser(pth)))


    #pth_chck = os.path.join("checkpoint", 'g_srgan.npz')
    #pth_dst = r"C:\Users\ksteinfe\Desktop\TEMP"
    #pth_src = r"C:\Users\ksteinfe\Desktop\TEST\superres_test"

    # create main parser
    parser = argparse.ArgumentParser()
    parser.add_argument('source_path', help="path at which to find source images. all JPG and PNG files less than {}px will be processed.".format(MAX_SIZE))
    parser.add_argument('destination_path', help="path at which to save upscaled images.")
    parser.add_argument('model_path', help="path to trained srgan model (*.npz)")
    args = parser.parse_args()

    upsample_images(args.source_path, args.destination_path, args.model_path)
