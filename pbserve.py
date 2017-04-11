import argparse
import os
import scipy.misc
import numpy as np

import tensorflow as tf

import web
import json

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

        # create web.py server
        urls = (
            '/run/local_id/(.*)', 'index'
        )
        webapp = serve(urls, globals())
        web.generated_result = generated
        web.sess = tf.Session()
        if config:
            webapp.run(port=config["port"])
        else:
            webapp.run(port=8081)

def image_overlay_linear(master, x, y, subimg, input_size):
    for i in range(3, input_size-5):
        for j in range(3, input_size-5):
            if (master[y + j, x + i] == -1.1):
                master[y + j, x + i] = subimg[j, i]
            else:
                if i < 64 and j < 64:
                    r = min(i, j)
                    master[y + j, x + i] = ((64-r)/64.0*master[y + j, x + i] + r/64.0*subimg[j, i])
                elif i < 64:
                    master[y + j, x + i] = ((64-i)/64.0*master[y + j, x + i] + i/64.0*subimg[j, i])
                elif j < 64:
                    master[y + j, x + i] = ((64-j)/64.0*master[y + j, x + i] + j/64.0*subimg[j, i])
                else:
                    master[y + j, x + i] = subimg[j, i]

def test_single_image_full(web, param):
    if config:
        input_size = config["input_size"]
    else:
        input_size = 256

    input_img = scipy.misc.imread('./test/input_{}.jpg'.format(param['id']), flatten = True).astype(np.float)
    h, w = input_img.shape
    pad_h = h+ input_size #(h+255) // 256 * 256
    pad_w = w + input_size #(w+255) // 256 * 256
    pad_img = np.pad(input_img[:h, :w], pad_width=((0, pad_h-h), (0, pad_w-w)), mode='constant')
    x = y = n = 0
    output_img = np.empty((pad_h, pad_w), np.float)
    output_img.fill(-1.1)
    while x < w:
        while y < h:
            print('x & y', x, y)
            sub_img = pad_img[y:y+input_size, x:x+input_size]
            sub_img = sub_img / 127.5 - 1.
            if (np.mean(sub_img) < - 0.999):
                generated = np.ones_like(sub_img)
            else:
                sample_image = np.empty((input_size, input_size, 6), dtype=np.float32)
                for c in range(6):
                    sample_image[:, :, c] = sub_img
                sample_images = [sample_image]
                generated = web.sess.run(web.generated_result,
                    feed_dict={'pix2pix/real_A_and_B_images:0': sample_images}
                )
            image_overlay_linear(output_img, x, y, np.squeeze(generated), input_size)
            n = n + 1
            y = y + input_size - 64
        x = x + input_size - 64
        y = 0
    output_crop_img = output_img[:h, :w]
    scipy.misc.imsave('./test/output_{}.jpg'.format(param['id']), output_crop_img)


def test_single_image(web, param):
    if config:
        input_size = config["input_size"]
    else:
        input_size = 256
    input_img = scipy.misc.imread('./test/input_{}.jpg'.format(param['id']), flatten = True).astype(np.float)
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
    scipy.misc.imsave('./test/output_{}.jpg'.format(param['id']), sample)

class index:
    def GET(self, param):
        print(param)
        test_single_image(web, param)
        return '200 OK'

    def POST(self, param):
        str = web.data().decode()
        print(str)
        data = json.loads(str)
        if config:
            if config["max_resolution"] > 512:
                test_single_image_full(web, data)
            else:
                test_single_image(web, data)
        else:
            test_single_image(web, data)
        return '200 OK'

class serve(web.application):
    def run(self, port=8081, *middleware):
        func = self.wsgifunc(*middleware)
        return web.httpserver.runsimple(func, ('0.0.0.0', port))

if __name__ == '__main__':
    tf.app.run()
