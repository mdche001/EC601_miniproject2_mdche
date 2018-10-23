import tensorflow as tf
import numpy as np
import os
config = tf.ConfigProto()
config.gpu_options.allow_growth = True
session = tf.Session(config=config)

# train_dir = 'D:/program/mini_project2/venv/data/train/'

def get_files(file_dir):
    "return list of images and labels"

    #initialize the image_list and label_list
    image_list = []
    label_list = []
    count_roses = 0
    count_sunflowers = 0

    #Building the image_list and lable_list
    for file in os.listdir(file_dir):
        name = file.split(sep='.')
        image_list.append(file_dir + file)
        if name[0] == 'rose':
            label_list.append(0)
            count_roses += 1
        else:
            label_list.append(1)
            count_sunflowers += 1
    # print('There are %d roses\n ' %(count_roses))
    # print('There are %d sunflowers ' %(count_sunflowers))

    image_list = np.asarray(image_list)
    label_list = np.asarray(label_list)

    #shuffle the list
    temp = np.array([image_list, label_list])
    temp = temp.transpose()
    np.random.shuffle(temp)

    image_list = list(temp[:,0])
    label_list = list(temp[:,1])
    print(label_list)
    label_list = [int(float(i)) for i in label_list]

    return image_list, label_list

def get_batch(image, label, image_W, image_H, batch_size, capacity):
    "return image_batch with [batch_size, width, height, 3], dtype = tf.float32  label.batch with[batch_size], dtype = tf.int32"
    '''
    :parameter:
        image,label: list from get
        image_W：image width
        image_H: image height
        batch_size: batch size
        capacity: queue size
    '''

    #convert to d-type
    image = tf.cast(image, tf.string)
    label= tf.cast(label, tf.int32)
    train_list = [image,label]
    #make input queue
    input_queue = tf.train.slice_input_producer(train_list, shuffle=False)

    #images tags
    label = input_queue[1]

    #load images
    image = tf.read_file(input_queue[0])
    image = tf.image.decode_jpeg(image, channels = 3)
    image = tf.image.resize_image_with_crop_or_pad(image, image_W, image_H)
    image = tf.image.per_image_standardization(image)
    image_batch, label_batch = tf.train.batch([image, label],
                                                batch_size= batch_size,
                                                num_threads= 64,
                                                capacity= capacity)

    #get batch
    label_batch = tf.reshape(label_batch, [batch_size])
    return  image_batch, label_batch


# test input
# import matplotlib.pyplot as plt
#
# BATCH_SIZE = 2
# CAPACITY = 256
# IMG_W = 208
# IMG_H = 208
#
# # test input directory
# # train_dir = 'D:/program/mini_project2/venv/data/test/'
# #train_dir = 'D:/program/mini_project2/venv/data/train/'
#
# train_dir = os.getcwd()+'/venv/data/train/'
# image_list, label_list = get_files(train_dir)
# image_batch, label_batch = get_batch(image_list, label_list, IMG_W, IMG_H, BATCH_SIZE, CAPACITY)
#
# with tf.Session() as sess:
#     i = 0
#     coord = tf.train.Coordinator()
#     threads = tf.train.start_queue_runners(coord= coord)
#
#     #security input from google guide
#     try:
#         while not coord.should_stop() and i < 1:
#             img, label = sess.run([image_batch, label_batch])
#
#             for j in np.arange(BATCH_SIZE):
#                 print('label: %d' %label[j])
#                 plt.imshow(img[j,:,:,:])
#                 plt.show()
#             i += 1
#     except tf.errors.OutOfRangeError:
#         print('done!')
#     finally:
#         coord.request_stop()
#     coord.join(threads)