import argparse
import os
import scipy.misc
import numpy as np

import tensorflow as tf

import web
import json

def main(_):
    print("GanMook serve!")
    if not os.path.exists('./test'):
        os.makedirs('./test')

    with tf.gfile.GFile('model/ganmook_clf_1.pb', 'rb') as f:
        graph_def = tf.GraphDef()
        graph_def.ParseFromString(f.read())

    print(tf.trainable_variables());

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
        webapp.run(port=8081)

def test_single_image(web, param):
    input_img = scipy.misc.imread('./test/input_{}.jpg'.format(param['id']), flatten = True).astype(np.float)
    h, w = input_img.shape
    input_img = scipy.misc.imresize(input_img, [256, 256])
    input_img = input_img/127.5 - 1.
    sample_image = np.empty((256, 256, 6), dtype=np.float32)
    for i in range(6):
        sample_image [:, :, i] = input_img
    sample_image = [sample_image]

    samples = web.sess.run(
        web.generated_result,
        feed_dict={'pix2pix/real_A_and_B_images:0': sample_image}
    )
    sample = scipy.misc.imresize(samples[0], [h, w])
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
        test_single_image(web, data)
        return '200 OK'

class serve(web.application):
    def run(self, port=8081, *middleware):
        func = self.wsgifunc(*middleware)
        return web.httpserver.runsimple(func, ('0.0.0.0', port))

if __name__ == '__main__':
    tf.app.run()
