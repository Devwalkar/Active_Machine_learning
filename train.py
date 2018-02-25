"""Train helper functions."""
import os
import random
from utils.visualization import map_to_patch
from keras.preprocessing.image import ImageDataGenerator, load_img, img_to_array
from numpy import array,zeros,float32,array_equal,unique,int8
from numpy.random import randint
from numpy import sum as npsum

def get_data_iterators(horizontal_flip=False, vertical_flip=False, width_shift_range=0.0,
                       height_shift_range=0.0, rotation_range=0, zoom_range=0.0,
                       batch_size=1, data_dir='data', target_size=(512, 512),
                       fill_mode='constant', rescale=1 / 255., load_train_data=False,
                       color_mode='rgb', seed=None):


    """Create data iterator."""
    # Reference : https://machinelearningmastery.com/image-augmentation-deep-learning-keras/
    aug_gen = ImageDataGenerator(horizontal_flip=horizontal_flip, vertical_flip=vertical_flip,
                                 width_shift_range=width_shift_range, height_shift_range=height_shift_range,
                                 rotation_range=rotation_range, zoom_range=zoom_range,
                                 fill_mode=fill_mode, rescale=rescale)
    data_gen = ImageDataGenerator(rescale=rescale)

    # Training data ...............
    if load_train_data:
        X_train, y_train = load_dataset(data_dir=os.path.join(data_dir, 'train'), target_size=target_size,
                                        color_mode=color_mode)
        # Data Augmentation for training data and data scaling for validation and test sets
        train_it = aug_gen.flow(X_train, y_train, batch_size=batch_size, seed=seed)

    else:
        train_it = aug_gen.flow_from_directory(os.path.join(data_dir, 'train'),
                                               batch_size=batch_size, target_size=target_size,
                                               class_mode='binary', color_mode=color_mode, seed=seed)
    # Validation data................

    if load_train_data:
        X_val, y_val = load_dataset(data_dir=os.path.join(data_dir, 'val'), target_size=target_size,
                                    color_mode=color_mode)
        val_it = data_gen.flow(X_val, y_val, batch_size=batch_size, seed=seed)
    else:
        val_it = data_gen.flow_from_directory(os.path.join(data_dir, 'val'),
                                              batch_size=batch_size, target_size=target_size,
                                              class_mode='binary', color_mode=color_mode, seed=seed)

    # Oracle data................
  
    if load_train_data:
        X_oracle, y_oracle = load_dataset(data_dir=os.path.join(data_dir, 'oracle'), target_size=target_size,
                                          color_mode=color_mode)
        oracle_it = data_gen.flow(X_oracle, y_oracle, batch_size=batch_size, seed=seed)
    else:
        oracle_it = data_gen.flow_from_directory(os.path.join(data_dir, 'oracle'),
                                                 batch_size=batch_size, target_size=target_size,
                                                 class_mode='binary', color_mode=color_mode, seed=seed)
   
    # hint : any .png image inside the directory given will be considered in flow_from_dictionary
    test_it = data_gen.flow_from_directory(os.path.join(data_dir, 'test'),
                                           batch_size=batch_size, target_size=target_size,
                                           class_mode='binary', color_mode=color_mode, seed=seed)

    return train_it,oracle_it,val_it, test_it


def get_lesion_iterators(horizontal_flip=False, vertical_flip=False, width_shift_range=0.,
                         height_shift_range=0., rotation_range=0, zoom_range=0.,
                         batch_size=1, base_dir='eoptha', img_dir='combined_cut',
                         annot_dir='Annotation_combined_cut', target_size=(512, 512),
                         fill_mode='constant', rescale=1 / 255., seed=None,
                         out_size=16, f_size=63):
    """Create data iterator."""
    aug_gen = ImageDataGenerator(horizontal_flip=horizontal_flip, vertical_flip=vertical_flip,
                                 width_shift_range=width_shift_range,
                                 height_shift_range=height_shift_range,
                                 rotation_range=rotation_range, zoom_range=zoom_range,
                                 fill_mode=fill_mode, rescale=rescale)
    data_gen = ImageDataGenerator(rescale=rescale)

    if seed is None:
        seed = random.randint(1, 1e6)

    def pair_iterator(img_it, annot_it):
        while True:
            x, _ = next(img_it)
            xa, _ = next(annot_it)

            y = zeros((xa.shape[0], 1, out_size, out_size), dtype=float32)
            for i, xai in enumerate(xa):
                y[i, 0] = annotation_to_lesion(xai[0], out_size=out_size, f_size=f_size)

            assert array_equal(unique(y), [0, 1]) or array_equal(unique(y), [0]), 'Something odd on the y array ({0})'.format(unique(y))

            yield x, y

    def create_iterator(gen, phase):
        img_it = gen.flow_from_directory(os.path.join(base_dir, img_dir, phase),
                                         batch_size=batch_size, target_size=target_size,
                                         color_mode='rgb', seed=seed)
        annot_it = gen.flow_from_directory(os.path.join(base_dir, annot_dir, phase),
                                           batch_size=batch_size, target_size=target_size,
                                           color_mode='grayscale', seed=seed)
        return pair_iterator(img_it, annot_it)

    train_it = create_iterator(aug_gen, 'train')
    val_it = create_iterator(data_gen, 'val')
    test_it = create_iterator(data_gen, 'test')

    return train_it, val_it, test_it


def annotation_to_lesion(annot, out_size=16, f_size=63, threshold=0, img_size=512):
    out = zeros((out_size, out_size), dtype=float32)
    for y in range(out_size):
        for x in range(out_size):
            yi, xi = map_to_patch(y, x, f_size=f_size)
            yf, xf = yi + f_size, xi + f_size

            xmi, ymi = (xi + f_size//4 - 1, yi + f_size//4 - 1)
            xmf, ymf = (xi + (f_size//4)*3 + 1, yi + (f_size//4)*3 + 1)

            if yi < 0:
                yi = 0
            if xi < 0:
                xi = 0
            if xmi < 0:
                xmi = 0
            if ymi < 0:
                ymi = 0
            if  xmf > img_size:
                xmf = img_size
            if ymf > img_size:
                ymf = img_size

            out[y, x] = npsum(annot[ymi:ymf, xmi:xmf]) > threshold
            # out[y, x] = npsum(annot[yi:yf, xi:xf]) > threshold

    return out


'''
load_dataset:   Loads the dataset into the memory
                imgs have the paths to the normal and pathological images in the called data directory
                y has their corresponding labels. for normal class 0 and for pathological class 1
                X will have the shape as (len(imgs),channels,512,512)
                returns X and y

'''



def load_dataset(data_dir='data', classes=['normal', 'pathological'], color_mode='rgb',
                 target_size=(512, 512)):
    """Load dataset into memory."""

    imgs = []
    y = []
    for i, c in enumerate(classes):
        c_imgs = os.listdir(os.path.join(data_dir, c))
        for c_img in c_imgs:
            if c_img.endswith('.png'):
                imgs.extend([os.path.join(data_dir, c, c_img)])
                y.extend([i])

    N = len(imgs)
    idx = list(range(N))

    #shuffling the data
    random.shuffle(idx)
    imgs = array(imgs)
    y = array(y, dtype=int8)
    imgs = imgs[idx]
    y = y[idx]

    # Color specifications
    grayscale = False
    channels = 3
    if color_mode == 'grayscale':
        grayscale = True
        channels = 1


    X = zeros((N, channels) + target_size, dtype=float32)
    for i, img_path in enumerate(imgs):
        img = load_img(img_path, grayscale=grayscale, target_size=target_size)
        #img is an object. so convert to array of X[i] of shape(3,512,512)
        X[i] = img_to_array(img)
    print('%s data loaded'%data_dir)
    return X, y
