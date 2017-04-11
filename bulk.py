import cv2
import argparse
import os
import scipy.misc
import numpy as np

import tensorflow as tf

def color_process(color_img, color_filter=True):
    hsv_img = cv2.cvtColor(color_img, cv2.COLOR_BGR2HSV)
    print(hsv_img.shape)
    for i in range(h-1):
        for j in range(w-1):
                hue, sat, val = hsv_img[i, j]
                if (color_filter):
                    if (hue > 100 and  hue < 170):
                        val = val * 0.7
                        sat = sat * 0.4
                    elif (hue> 90 and hue < 100):
                        sat = sat * 0.7
                        val = val * 0.9
                else:
                    hue = int(hue / 3.0) * 3.0
                    sat = min(255, sat * 1.1)

                hsv_img[i,j] = [hue, sat, val]

    color_img = cv2.cvtColor(hsv_img, cv2.COLOR_HSV2BGR)
    cv2.imwrite('temp.jpg', color_img)
    return color_img

parser = argparse.ArgumentParser(description='')
parser.add_argument('--config_file', dest='config_file', default='serve.json', help='name of the config file')
args = parser.parse_args()

if not os.path.exists('./test'):
    os.makedirs('./test')
if os.path.exists(args.config_file):
    with open(args.config_file) as data_file:
        config = json.load(data_file)
else:
    print(args.config_file, 'is not found. Using default config.')
    config = []

def main(_):
    print("GanMook serve!")

    if config:
        pb_file = config["model_file"]
    else:
        pb_file = 'model/ganmook_clf_2.pb'

    with tf.gfile.GFile(pb_file, 'rb') as f:
        graph_def = tf.GraphDef()
        graph_def.ParseFromString(f.read())
    print('Loading ', pb_file)

    with tf.Graph().as_default() as graph:
        generated,  = tf.import_graph_def(
            graph_def,
            input_map = None,
            return_elements = ['generator/Tanh:0'],
            name = 'pix2pix'
        )

    if config:
        input_size = config["input_size"]
    else:
        input_size = 256

    for n in range(1024):
        input_img = scipy.misc.imread('./test/input_{}.jpg'.format(n), flatten = True).astype(np.float)
        h, w = input_img.shape
        input_img = scipy.misc.imresize(input_img, [input_size, input_size])
        input_img = input_img/127.5 - 1.
        sample_image = np.empty((input_size, input_size, 6), dtype=np.float32)
        for i in range(6):
            sample_image [:, :, i] = input_img
        sample_images = [sample_image]

        samples = web.sess.run(
            web.generated_result,
            feed_dict={'pix2pix/real_A_and_B_images:0': sample_images}
        )
        sample = np.squeeze(samples[0])
        sample = scipy.misc.imresize(sample, [h, w])
        scipy.misc.imsave('./test/output_{}.jpg'.format(n), sample)

        img = cv2.imread('./test/output_{}.jpg'.format(n))
        img = color_process(color_img, color_filter=False)
        cv2.imwrite('./test/output_{}.jpg'.format(n), img)

if __name__ == '__main__':
    tf.app.run()
