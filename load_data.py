import os
from PIL import Image

import numpy as np

img_size = 28
labels = ['dog', 'cat']

def load_train_data_from_jpg():
    train = dict()
    for label in labels:
        print('Load', label, 'images.')
        imgs = []
        for i in range(12500):
            fname = 'train/{label}.{i}.jpg'.format(label=label, i=i)
            img = Image.open(fname)
            img = img.resize((img_size, img_size))
            img = np.array(img, np.float32)
            imgs.append(img)
            if i % 1000 == 999:
                print('{num} images loaded.'.format(num=i+1))
        imgs = np.array(imgs)
        train.update({label: imgs})
    np.savez('train.npz', dog=train['dog'], cat=train['cat'])
    return train['dog'], train['cat']

def load_train_data_from_npz():
    with np.load('train.npz') as npz:
        train = npz['dog'], npz['cat']
    return train

def load_test_data_from_jpg():
    imgs = []
    for i in range(12500):
        fname = 'test/{i}.jpg'.format(i=i+1)
        img = Image.open(fname)
        img = img.resize((img_size, img_size))
        img = np.array(img, np.float32)
        imgs.append(img)
        if i % 1000 == 999:
            print('{num} images loaded.'.format(num=i+1))
    imgs = np.array(imgs)
    np.savez('test.npz', images=imgs)
    return imgs

def load_test_data_from_npz():
    with np.load('test.npz') as npz:
        test = npz['images']
    return test

def load_data():
    try:
        train = load_train_data_from_npz()
    except Exception as e:
        print(e)
        train = load_train_data_from_jpg()
    try:
        test = load_test_data_from_npz()
    except Exception as e:
        print(e)
        test = load_test_data_from_jpg()
    return train, test
