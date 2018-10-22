import tensorflow as tf
import numpy as np
import os

# img_width = 208
# img_height = 208

# train_dir = 'D:/program/mini_project2/venv/data/train/'

def get_files(file_dir):
    "return list of images and labels"
    roses = []
    label_roses = []
    sunflowers = []
    label_sunflowers = []
    for file in os.listdir(file_dir):
        name = file.split(sep='.')
        if name[0] == 'rose':
            roses.append(file_dir + file)
            label_roses.append(0)
        else:
            sunflowers.append(file_dir + file)
            label_sunflowers.append(1)
    print('There are %d roses\n There are %d sunflowers' %(len(roses), len(sunflowers)))

    image_list = np.hstack((roses, sunflowers))
    label_list = np.hstack((label_roses, label_sunflowers))

    temp = np.array([image_list, label_list])
    temp = temp.transpose()
    np.random.shuffle(temp)

    image_list = list(temp[:,0])
    label_list = list(temp[:,1])
    print(label_list)
    label_list = [int(float(i)) for i in label_list]

    return image_list, label_list

def get_batch(image, label, image_W, image_H, batch_size, capacity):
    "return image_batch: 4D tensor[batch_size, width, height, 3], dtype = tf.float32  label.batch: 1D tensor[batch_size], dtype = tf.int32"
    '''
    args:
        image: list type
        label: list type
        image_W： image width
        image_H: image height
        batch_size: batch size
        capacity: the maximum elements in queue
    '''
    image = tf.cast(image, tf.string)
    label= tf.cast(label, tf.int32)

    #make input queue
    input_queue = tf.train.slice_input_producer([image, label])

    label = input_queue[1]
    image_contents = tf.read_file(input_queue[0])
    image = tf.image.decode_jpeg(image_contents, channels = 3)
    #
    # data argumentation
    #
    image = tf.image.resize_image_with_crop_or_pad(image, image_W, image_H)
    image = tf.image.per_image_standardization(image)
    image_batch, label_batch = tf.train.batch([image, label],
                                                batch_size= batch_size,
                                                num_threads= 64,
                                                capacity= capacity)

    # image_batch, label_batch = tf.train.shuffle_batch([image, label],
    #                                           batch_size=BATCH_SIZE,
    #                                           num_threads=64,
    #                                           capacity=CAPACITY,
    #                                           min_after_dequeue = CAPACITY - 1)

    label_batch = tf.reshape(label_batch, [batch_size])

    return  image_batch, label_batch


#test
# import matplotlib.pyplot as plt
#
# BATCH_SIZE = 2
# CAPACITY = 256
# IMG_W = 208
# IMG_H = 208
# train_dir = 'D:/program/mini_project2/venv/data/test/'
# #train_dir = 'D:/program/mini_project2/venv/data/train/'
#
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