import os
import random
import cv2
import re
import numpy as np
import pandas as pd
import tensorflow as tf
from matplotlib import pyplot as plt
from scipy.sparse import csr_matrix
from scipy.sparse import vstack

image_path = '../input/flickr30k_images/flickr30k_images/flickr30k_images/'

train_images_list = os.listdir(image_path)

sample_size = 30
train_images_list = train_images_list[:sample_size]

size = (256, 256)
num_channels = 3

train = np.array([None] * sample_size)
real_images = np.array([None] * sample_size)

j = 0
for i in train_images_list:
    train[j] = np.array(plt.imread(image_path + i))
    real_images[j] = np.array(plt.imread(image_path + i))
    j += 1

j = 0
for i in train:
    train[j] = cv2.resize(i, size)
    train[j] = train[j].reshape(1, size[0], size[1], num_channels)
    j += 1

train = np.vstack(train[:])

plt.imshow(np.squeeze(train[0]))
plt.show()

train_captions = pd.read_csv(
    '../input/flickr30k_images/flickr30k_images/results.csv', delimiter='|')

def get_images_id(names):
    names = [int(x.split('_')[-1].split('.')[0]) for x in names]
    return names

train_captions.columns = ['image_name', 'comment_number', 'comment']

def images_map_caption(train_images_list, train_captions):
    caption = []
    for i in train_images_list:
        caption.append(train_captions[train_captions['image_name'] == i]['comment'].iat[0])
    return caption

captions = np.array(images_map_caption(train_images_list, train_captions))
print(captions.shape)

start_tag = '<s>'
end_tag = '<e>'

def get_vocab(captions):
    arr = []
    m = captions.shape[0]
    sentence = [None ] * m
    j  = 0
    for i in captions:
        i = re.sub(' +',' ',i)
        i = start_tag + ' ' + i + ' ' + end_tag
        sentence[j] = i.split()
        j += 1
        arr = arr + i.split()
    arr = list(set(arr))
    vocab_size = len(arr)
    j = 0
    fwd_dict = {}
    rev_dict = {}
    j = 0
    for i in arr:
        fwd_dict[i] = j
        rev_dict[j] = i
        j += 1
    return vocab_size, sentence, fwd_dict, rev_dict

vocab_size, sentences, fwd_dict, rev_dict = get_vocab(captions)

m = len(sentences)
train_caption = [None] * m
i = 0
for sentence in sentences:
    cap_array = None
    for word in sentence:
        row = [0]
        col = [fwd_dict[word]]
        data = [1]
        if cap_array is None:
            cap_array = csr_matrix((data, (row, col)), shape=(1, vocab_size))
        else:
            cap_array = vstack((cap_array, csr_matrix((data, (row, col)), shape=(1, vocab_size))))
    train_caption[i] = cap_array
    i += 1


train_caption[0].shape