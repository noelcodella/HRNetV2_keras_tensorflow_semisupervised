# Noel C. F. Codella
# Example Semantic Segmentation Code for Keras / TensorFlow

# GLOBAL DEFINES
# Image dimensions and random seed
T_G_WIDTH = 448 #224
T_G_HEIGHT = 448 #224
T_G_NUMCHANNELS = 3
T_G_SEED = 1337

# How many images to load from disk, and how many
# before a print command to report progress
T_G_CHUNKSIZE = 1000
T_G_REPORTSIZE = 50

# when to save checkpoints (only one always overwritten)
T_G_CHECKPOINT = 10

USAGE_LEARN = 'Usage: \n\t -learn <Train Images (TXT)> <Train Masks (TXT)> <Val Images (TXT)> <Val Masks (TXT)> <color map (TXT)> <batch size> <num epochs> <output model prefix> <Unlabeled Images (TXT)> <option: load weights from...> \n\t -extract <Model Prefix> <Input Image List (TXT)> <Output Folder> \n\t\tScores a model '

# Misc. Necessities
import sys
import os
import ssl # these two lines solved issues loading pretrained model
ssl._create_default_https_context = ssl._create_unverified_context
import numpy as np
import matplotlib.pyplot as plt
import cv2
from scipy.misc import imresize
np.random.seed(T_G_SEED)
import scipy
from scipy.special import softmax

# TensorFlow Includes
import tensorflow as tf
tf.set_random_seed(T_G_SEED)

# Keras Imports & Defines 
import keras
import keras.applications
import keras.optimizers
import keras.losses
from keras import backend as K
from keras.models import Model
from keras import optimizers
import keras.layers as kl

from keras.preprocessing.image import ImageDataGenerator

# Local Imports
from model.seg_hrnet import seg_hrnet


#sys.stdout = open('./seg_output.log', 'w')

# Uncomment to use the TensorFlow Debugger
#from tensorflow.python import debug as tf_debug
#sess = K.get_session()
#sess = tf_debug.LocalCLIDebugWrapperSession(sess)
#K.set_session(sess)

# Generator object for data augmentation.
# Can change values here to affect augmentation style.
datagen = ImageDataGenerator(  rotation_range=20,
                                width_shift_range=0.05,
                                height_shift_range=0.05,
                                zoom_range=0.10,
                                horizontal_flip=True,
                                vertical_flip=True,
                                )


# A binary jaccard (non-differentiable)
def jaccard_index_b(y_true, y_pred):

    safety = 0.001

    y_true_f = K.cast(K.greater(K.flatten(y_true),0.5),'float32')
    y_pred_f = K.cast(K.greater(K.flatten(y_pred),0.5),'float32')

    top = K.sum(K.minimum(y_true_f, y_pred_f))
    bottom = K.sum(K.maximum(y_true_f, y_pred_f))

    return top / (bottom + safety)


# A binary jaccard (non-differentiable)
def jaccard_loss_b(y_true, y_pred):

    return 1 - jaccard_index_b(y_true, y_pred)

# An example loss based on multiple metrics
def joint_loss(y_true, y_pred):

    return 0.4 * jaccard_loss_b(y_true, y_pred) + 0.2 * soft_jaccard_loss(y_true, y_pred) + 0.2 * jaccard_loss(y_true, y_pred) + 0.2 * keras.losses.mean_squared_error(y_true, y_pred)

# A computation of the jaccard index
def jaccard_index(y_true, y_pred):
    
    safety = 0.001

    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)

    top = K.sum(K.minimum(y_true_f, y_pred_f))
    bottom = K.sum(K.maximum(y_true_f, y_pred_f))

    return top / (bottom + safety)

# An example loss based on jaccard index
def jaccard_loss(y_true, y_pred):

    return 1 - jaccard_index(y_true, y_pred)


# a 'soft' version of the jaccard index
def soft_jaccard_index(y_true, y_pred):

    safety = 0.001

    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)

    top = K.sum(y_true_f * y_pred_f)
    bottom = K.sum(y_true_f * y_true_f) + K.sum(y_pred_f * y_pred_f) - top

    return top / (bottom + safety)


def soft_jaccard_loss(y_true, y_pred):

    return 1 - soft_jaccard_index(y_true, y_pred)


# converts a colored image array to a class image array
def colorsToClass(clabels, cmap):

    numk = cmap.shape[0]
    karray = np.zeros((clabels.shape[0], clabels.shape[1], clabels.shape[2], numk))
    
    ncmap = cmap 

    # move flip outside of loop for efficiency (BGR -> RGB)
    clabels = np.flip(clabels, axis=3)

    for k in range(0, clabels.shape[0]):
        if (k % T_G_REPORTSIZE == 0):
                print "["+str(k)+"]...",
                sys.stdout.flush()

        for i in range(0, clabels.shape[1]): 
            for j in range(0, clabels.shape[2]):
                # BGR channel order, so flip
                #c = np.flip(np.squeeze(clabels[k,i,j,:]))
                c = np.squeeze(clabels[k,i,j,:])
                
                # allow some tolerance in color matching
                kmap = (np.abs(ncmap-c)).sum(axis=1) < 20
                kmap = kmap.astype('float')

                karray[k,i,j,:] = kmap

    print "\n"

    return karray

def classToColors(clabels, cmap):

    numk = cmap.shape[0]
    carray = np.zeros((clabels.shape[0], clabels.shape[1], clabels.shape[2], 3))

    for k in range(0, clabels.shape[0]):
        if (k % T_G_REPORTSIZE == 0):
                print "["+str(k)+"]...",
                sys.stdout.flush()

        for i in range(0, clabels.shape[1]):
            for j in range(0, clabels.shape[2]):
                c = np.argmax(np.squeeze(clabels[k,i,j,:]))

                pix = cmap[c]
                
                # BGR channel order, so flip
                carray[k,i,j,:] = np.flip(pix)

    print "\n"

    return carray



# generator function for data augmentation
def createDataGen(X, Y, b, cmap):

    local_seed = T_G_SEED
    genX = datagen.flow(X,Y, batch_size=b, seed=local_seed, shuffle=False)
    genY = datagen.flow(Y,Y, batch_size=b, seed=local_seed, shuffle=False)
    while True:
            Xi = genX.next()
            Yi = genY.next()

            yield Xi[0], Yi[0]



def createModel(batch_size, height, width, channel, classes, namepref='model'):

    base_model = seg_hrnet(batch_size, height, width, channel, classes, namepref)

    print base_model.summary()

    base_model.compile(optimizer=keras.optimizers.Adam(), loss=soft_jaccard_loss, metrics=[jaccard_loss_b, keras.losses.binary_crossentropy, joint_loss, keras.losses.mean_squared_error, soft_jaccard_loss, jaccard_loss])

    return base_model


def t_save_image_list(inputimagelist, start, length, pred, outputpath, rdim=T_G_WIDTH):

    # Count the number of images in the list
    list_file = open(inputimagelist, "r")
    content = list_file.readlines()
    content = content[start:start+length]

    c_img = 0
    for img_file in content:
        img_file = img_file.rstrip('\n')
        filename = outputpath + "/" + img_file
        if not os.path.exists(os.path.dirname(filename)):
            os.makedirs(os.path.dirname(filename))
        outimg = (pred[c_img,:,:,:])*255.
        cv2.imwrite(filename, outimg)
        c_img = c_img + 1


# loads an image and preprocesses
def t_read_image(loc, prep=1):
    t_image = cv2.imread(loc)
    t_image = cv2.resize(t_image, (T_G_HEIGHT,T_G_WIDTH))
    t_image = t_image.astype("float32")

    if (prep == 1):
        t_image = keras.applications.densenet.preprocess_input(t_image, data_format='channels_last')

    return t_image

def t_norm_image(img):
    new_img = (img - np.amin(img)) / (np.amax(img) - np.amin(img))

    return new_img

# loads a set of images from a text index file   
def t_read_image_list(flist, start, length, color=1, norm=0, prep=1):

    with open(flist) as f:
        content = f.readlines() 
    content = [x.strip().split()[0] for x in content] 

    datalen = length
    if (datalen < 0):
        datalen = len(content)

    if (start + datalen > len(content)):
        datalen = len(content) - start

    if (color == 1):
        imgset = np.zeros((datalen, T_G_HEIGHT, T_G_WIDTH, T_G_NUMCHANNELS))
    else:
        imgset = np.zeros((datalen, T_G_HEIGHT, T_G_WIDTH, 1))

    for i in range(start, start+datalen):
        if ((i-start) < len(content)):
            val = t_read_image(content[i], prep)
            if (color == 0):
                val = val[:,:,0]
                val = np.expand_dims(val,2)
            imgset[i-start] = val
            if (norm == 1):
                imgset[i-start] = (t_norm_image(imgset[i-start]) * 1.0 + 0.0) 
            if (i % T_G_REPORTSIZE == 0):
                print "("+str(i)+")...",
                sys.stdout.flush()

    print("\n")            

    return imgset


def file_numlines(fn):
    with open(fn) as f:
        return sum(1 for _ in f)


def main(argv):

    if len(argv) < 2:
        print USAGE_LEARN    
        return

    if 'learn' in argv[0]:
        learn(argv[1:])
    elif 'extract' in argv[0]:
        extract(argv[1:])    

    return


def extract(argv):

    if len(argv) < 3:
        print 'Usage: \n\t <Model Prefix> <Input Image List (TXT)> <Output Path> \n\t\tExtracts model'
        return

    modelpref = argv[0]
    imglist = argv[1]
    outfile = argv[2]

    sys.stdout = open(outfile + '_train_output.log', 'w')
    print 'Command line: ',
    for a in argv:
        print a+' ',
    print ' '

    with open(modelpref + '.json', "r") as json_file:
        model_json = json_file.read()

    loaded_model = keras.models.model_from_json(model_json)
    loaded_model.load_weights(modelpref + '.h5')
    cmap = np.loadtxt(modelpref + '.cmap.txt')

    base_model = loaded_model 

    scoreModel(imglist,base_model,outfile,cmap)

    return

def scoreModel(imglist, base_model, outfile, cmap, aug=1):

    chunksize = T_G_CHUNKSIZE
    total_img = file_numlines(imglist)
    total_img_ch = int(np.ceil(total_img / float(chunksize)))

    for i in range(0, total_img_ch):
        imgs = t_read_image_list(imglist, i*chunksize, chunksize)
        valsa = base_model.predict(imgs)
        
        # test time data augmentation
        if (aug > 0):
            valsb = base_model.predict(scipy.ndimage.rotate(imgs, 90, axes=(2,1), reshape=False))
            valsb = scipy.ndimage.rotate(valsb, 270, axes=(2,1), reshape=False)
            valsc = base_model.predict(scipy.ndimage.rotate(imgs, 180, axes=(2,1), reshape=False))
            valsc = scipy.ndimage.rotate(valsc, 180, axes=(2,1), reshape=False)
            valsd = base_model.predict(scipy.ndimage.rotate(imgs, 270, axes=(2,1), reshape=False))
            valsd = scipy.ndimage.rotate(valsd, 90, axes=(2,1), reshape=False)
            #valse = base_model.predict(np.roll(imgs, 10, axis=2))
            #valse = np.roll(valse, -10, axis=2)
            #valsf = base_model.predict(np.roll(imgs, 10, axis=1))
            #valsf = np.roll(valsf, -10, axis=1)

            vals = (valsa + valsb + valsc + valsd) / 4.0

        else:
            vals = valsa

        valsout = classToColors(vals, cmap) / 255.

        t_save_image_list(imglist, i*chunksize, chunksize, valsout, outfile)

    return



def learn(argv):
    
    if len(argv) < 8:
        print USAGE_LEARN
        return

    # training images / maps
    in_t_i = argv[0]
    in_t_m = argv[1]

    # validation images / maps
    in_v_i = argv[2]
    in_v_m = argv[3]

    # color map file (label to color)
    cmapfile = argv[4]

    # controllable training parameters 
    batch = int(argv[5])
    numepochs = int(argv[6])
    outpath = argv[7] 

    # unlabeled data for semi-supervised consistency loss
    in_c_i = argv[8]

    sys.stdout = open(outpath + '_train_output.log', 'w')
    print 'Command line: ',
    for a in argv:
        print a+' ',
    print ' '

    # load the class assignment color map
    cmap = np.loadtxt(cmapfile)
    numk = cmap.shape[0]

    # chunksize is the number of images we load from disk at a time
    chunksize = T_G_CHUNKSIZE
    total_t = file_numlines(in_t_i)
    total_v = file_numlines(in_v_i)
    total_c = file_numlines(in_c_i)
    total_t_ch = int(np.ceil(total_t / float(chunksize)))
    total_v_ch = int(np.ceil(total_v / float(chunksize)))
    total_c_ch = int(np.ceil(total_c / float(chunksize)))

    print 'Dataset has ' + str(total_t) + ' training, and ' + str(total_v) + ' validation.'

    print 'Creating a model ...'
    model = createModel(batch, T_G_HEIGHT, T_G_WIDTH, T_G_NUMCHANNELS, numk, outpath)

    if len(argv) > 9:
        print 'Loading weights from: ' + argv[9] + ' ... '
        model.load_weights(argv[9], by_name=True)

    print 'Training loop ...'
   
    images_t = []
    masks_t = []
    images_v = []
    masks_v = []
    images_c = []

    t_imloaded = 0
    v_imloaded = 0
    c_imloaded = 0
 
    # manual loop over epochs to support very large sets 
    for e in range(0, numepochs):

        # training loop
        for t in range(0, total_t_ch):

            print 'Epoch ' + str(e) + ': train chunk ' + str(t+1) + '/ ' + str(total_t_ch) + ' ...'

            if ( t_imloaded == 0 or total_t_ch > 1 ): 
                print 'Reading image lists ...'
                images_t = t_read_image_list(in_t_i, t*chunksize, chunksize)
                colors_t = t_read_image_list(in_t_m, t*chunksize, chunksize, 1, 0, 0)

                print 'Parsing image labels ...'
                masks_t = colorsToClass(colors_t, cmap) 
                t_imloaded = 1

            print 'Starting to fit ...'

            # This method uses data augmentation
            model.fit_generator(generator=createDataGen(images_t,masks_t,batch,cmap), steps_per_epoch=len(images_t) / batch, epochs=1, shuffle=False, use_multiprocessing=True)
        
        # Consistency Loss
        for c in range(0, total_c_ch):

            print 'Epoch ' + str(e) + ': consistency chunk ' + str(c+1) + '/ ' + str(total_c_ch) + ' ...'

            if ( c_imloaded == 0 or total_c_ch > 1 ): 
                print 'Reading consistency image lists ...'
                images_c = t_read_image_list(in_c_i, c*chunksize, chunksize)
                c_imloaded = 1

            print 'Scoring unlabeled data ...'
            sys.stdout.flush()
            masks_c = model.predict(images_c)

            print 'Re-training on augmentations of scored images ...'
            sys.stdout.flush()
            # apply softmax to output scores to prefer higher scores
            # masks_c = softmax(masks_c, axis=3)
            masks_c = np.power(masks_c, 2.0)

            print 'Fitting ...'
            sys.stdout.flush()
            # Data Generator will equally perturb images as well as network predictions
            model.fit_generator(generator=createDataGen(images_c,masks_c,batch,cmap), steps_per_epoch=len(images_c) / batch, epochs=1, shuffle=False, use_multiprocessing=True)
        

        # Validation loop
        # In case the validation images don't fit in memory, we load chunks from disk again. 
        val_res = [0.0, 0.0]
        total_w = 0.0
        for v in range(0, total_v_ch):

            print 'Epoch ' + str(e) + ': val chunk ' + str(v+1) + '/ ' + str(total_v_ch) + ' ...'

            if ( v_imloaded == 0 or total_v_ch > 1 ):
                print 'Loading validation image lists ...'
                images_v = t_read_image_list(in_v_i, v*chunksize, chunksize)
                colors_v = t_read_image_list(in_v_m, v*chunksize, chunksize, 1, 0, 0)
                masks_v = colorsToClass(colors_v, cmap)
                v_imloaded = 1

            # Weight of current validation measurement. 
            # if loaded expected number of items, this will be 1.0, otherwise < 1.0, and > 0.0.
            w = float(images_v.shape[0]) / float(chunksize)


            curval = model.evaluate(images_v, masks_v, batch_size=batch)
            val_res[0] = val_res[0] + w*curval[0]
            val_res[1] = val_res[1] + w*curval[1]
            total_w = total_w + w

        val_res = [x / total_w for x in val_res]

        print 'Validation Results: ' + str(val_res)

        if (e % T_G_CHECKPOINT == 0):

            print "Saving Checkpoint ..."
            np.savetxt(outpath + '.cmap.txt',cmap)
            model.save(outpath + '.checkpoint.h5')
            model_json = model.to_json()
            with open(outpath + '.json', "w") as json_file:
                json_file.write(model_json)
                json_file.close()


    print 'Saving model ...'

    # save the color map
    np.savetxt(outpath + '.cmap.txt',cmap)

    # Save the model and weights
    model.save(outpath + '.h5')

    # Due to some remaining Keras bugs around loading custom optimizers
    # and objectives, we save the model architecture as well
    model_json = model.to_json()
    with open(outpath + '.json', "w") as json_file:
        json_file.write(model_json)
        json_file.close()

    # scoreModel(in_v_i, model, 'debugout')

    return


# Main Driver
if __name__ == "__main__":
    main(sys.argv[1:])
