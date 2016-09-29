import os
from PIL import Image

import numpy as np

labels = ['dog', 'cat']

def load_train_data_from_jpg(img_size):
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
    np.savez('train.{size}.npz'.format(size=img_size),
             dog=train['dog'], cat=train['cat'])
    return train['dog'], train['cat']

def load_train_data_from_npz(img_size):
    with np.load('train.{size}.npz'.format(size=img_size)) as npz:
        train = npz['dog'], npz['cat']
    return train

def load_test_data_from_jpg(img_size):
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
    np.savez('test.{size}.npz'.format(size=img_size), images=imgs)
    return imgs

def load_test_data_from_npz(img_size):
    with np.load('test.{size}.npz'.format(size=img_size)) as npz:
        test = npz['images']
    return test

def load_data(img_size):
    try:
        train = load_train_data_from_npz(img_size)
    except Exception as e:
        print(e)
        train = load_train_data_from_jpg(img_size)
    try:
        test = load_test_data_from_npz(img_size)
    except Exception as e:
        print(e)
        test = load_test_data_from_jpg(img_size)
    return train, test
