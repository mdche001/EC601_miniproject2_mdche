import os
import numpy as np
import tensorflow as tf
import input_data
import model
# import matplotlib.pyplot as plt

N_CLASSES = 2
IMG_W = 200
IMG_H = 200
BATCH_SIZE = 16
CAPACITY = 200
MAX_STEP = 3000
learning_rate = 0.0001

train_dir = 'D:/program/mini_project2/venv/data/train/'
test_dir = 'D:/program/mini_project2/venv/data/test/'
logs_train_dir = 'D:/program/mini_project2/venv/logs/train/'

def run_training():
    train, train_label = input_data.get_files(train_dir)
    train_batch, train_label_batch = input_data.get_batch(train,
                                                          train_label,
                                                          IMG_W,
                                                          IMG_H,
                                                          BATCH_SIZE,
                                                          CAPACITY)
    train_logits = model.inference(train_batch, BATCH_SIZE, N_CLASSES)
    train_loss = model.losses(train_logits, train_label_batch)
    train_op = model.training(train_loss, learning_rate)
    train_acc = model.evaluation(train_logits, train_label_batch)

    summary_op = tf.summary.merge_all()
    sess = tf.Session()
    train_writer = tf.summary.FileWriter(logs_train_dir, sess.graph)
    saver = tf.train.Saver()

    sess.run(tf.global_variables_initializer())
    coord = tf.train.Coordinator()
    threads = tf.train.start_queue_runners(sess=sess, coord=coord)

    try:
        for step in np.arange(MAX_STEP):
            if coord.should_stop():
                break
            _, tra_loss, tra_acc = sess.run([train_op, train_loss, train_acc])

            if step % 50 == 0:
                print('Step %d, train loss = %.2f, train accuracy = %.2f%%' %(step, tra_loss, tra_acc))
                summary_str = sess.run(summary_op)
                train_writer.add_summary(summary_str, step)

            if step % 200 == 0 or (step + 1) == MAX_STEP:
                chechpoint_path = os.path.join(logs_train_dir,'model.ckpt')
                saver.save(sess, chechpoint_path, global_step=step)

    except tf.errors.OutOfRangeError:
        print('Done training -- epoch limit reached')
    finally:
        coord.request_stop()

    coord.join(threads)
    sess.close()

#%% Evaluate one image when training, comment the following codes.

# from PIL import Image
# import matplotlib.pyplot as plt
# train_dir = 'D:/program/mini_project2/venv/data/train/'
# logs_train_dir = 'D:/program/mini_project2/venv/logs/train/'
#
# def get_one_image(file_dir):
#     """
#     Randomly pick one image from test data
#     Return: ndarray
#     """
#
#     test =[]
#     for file in os.listdir(file_dir):
#         test.append(file_dir + file)
#     print('There are %d test pictures\n' %(len(test)))
#
#     n = len(test)
#     ind = np.random.randint(0, n)
#     print(ind)
#     img_test = test[ind]
#
#     image = Image.open(img_test)
#     plt.imshow(image)
#     # plt.show()
#     image = image.resize([208, 208])
#     image = np.array(image)
#     return image
#
# test_dir = 'D:/program/mini_project2/venv/data/test/'
# def test_one_image():
#     """
#     Test one image with the saved models and parameters
#     """
#
#     test_image = get_one_image(test_dir)
#
#     with tf.Graph().as_default():
#         BATCH_SIZE = 1
#         N_CLASSES = 2
#
#         image = tf.cast(test_image, tf.float32)
#         image = tf.image.per_image_standardization(image)
#         image = tf.reshape(image, [1, 208, 208, 3])
#         logit = model.inference(image, BATCH_SIZE, N_CLASSES)
#
#         logit = tf.nn.softmax(logit)
#
#         x = tf.placeholder(tf.float32, shape=[208, 208, 3])
#
#         saver = tf.train.Saver()
#
#         with tf.Session() as sess:
#
#             print("Reading checkpoints...")
#             ckpt = tf.train.get_checkpoint_state(logs_train_dir)
#             if ckpt and ckpt.model_checkpoint_path:
#                 global_step = ckpt.model_checkpoint_path.split('/')[-1].split('-')[-1]
#                 saver.restore(sess, ckpt.model_checkpoint_path)
#                 print('Loading success, global_step is %s' % global_step)
#             else:
#                 print('No checkpoint file found')
#
#             prediction = sess.run(logit, feed_dict={x: test_image})
#             max_index = np.argmax(prediction)
#             if max_index==0:
#                 print('This is a daisy with possibility %.6f' %prediction[:, 0])
#             else:
#                 print('This is a rose with possibility %.6f' %prediction[:, 1])
#
#         plt.show()


def evaluate_all_image():
    '''
    Test all image against the saved models and parameters.
    Return global accuracy of test_image_set
    ##############################################
    ##Notice that test image must has label to compare the prediction and real
    ##############################################
    '''
    # you need to change the directories to yours.
    N_CLASSES = 2
    print('-------------------------')
    test, test_label = input_data.get_files(test_dir)
    BATCH_SIZE = len(test)
    print('There are %d test images totally..' % BATCH_SIZE)
    print('-------------------------')
    test_batch, test_label_batch = input_data.get_batch(test,
                                                        test_label,
                                                        IMG_W,
                                                        IMG_H,
                                                        BATCH_SIZE,
                                                        CAPACITY)
    logits = model.inference(test_batch, BATCH_SIZE, N_CLASSES)
    testloss = model.losses(logits, test_label_batch)
    testacc = model.evaluation(logits, test_label_batch)
    saver = tf.train.Saver()
    with tf.Session() as sess:
        print("Reading checkpoints...")
        ckpt = tf.train.get_checkpoint_state(logs_train_dir)
        if ckpt and ckpt.model_checkpoint_path:
            global_step = ckpt.model_checkpoint_path.split('/')[-1].split('-')[-1]
            saver.restore(sess, ckpt.model_checkpoint_path)
            print('Loading success, global_step is %s' % global_step)
        else:
            print('No checkpoint file found')
        print('-------------------------')
        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(sess=sess, coord=coord)
        test_loss, test_acc = sess.run([testloss, testacc])
        print('The model\'s loss is %.2f' % test_loss)
        correct = int(BATCH_SIZE * test_acc)
        print('Correct : %d' % correct)
        print('Wrong : %d' % (BATCH_SIZE - correct))
        print('The accuracy in test images are %.2f%%' % (test_acc * 100.0))
    coord.request_stop()
    coord.join(threads)
    sess.close()

if __name__ == '__main__':
    evaluate_all_image()