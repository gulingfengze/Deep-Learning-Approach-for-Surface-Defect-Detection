import os
import glob
import time
import argparse
from PIL import Image
import cv2 as cv
import numpy as np
import tensorflow as tf


'''
    Predict_Demo

'''

def concatImage(images,mode="L"):
    if not isinstance(images, list):
        raise Exception('images must be a  list  ')
    count=len(images)
    size= Image.fromarray(images[0]).size
    target = Image.new(mode, (size[0] * count, size[1] * 1))
    for i  in  range(count):
        image = Image.fromarray(images[i]).resize(size, Image.BILINEAR)
        target.paste(image, (i*size[0], 0, (i+1)*size[0], size[1]))
    return target


# Read the graph.
with tf.gfile.FastGFile('./checkpoint/frozen_inference_graph.pb', 'rb') as f:
    graph_def = tf.GraphDef()
    graph_def.ParseFromString(f.read())

config = tf.ConfigProto(allow_soft_placement=True,log_device_placement=False)
#config.gpu_options.per_process_gpu_memory_fraction = 0.8
config.gpu_options.allow_growth = True

with tf.Session(config=config) as sess:
    # Restore session
    sess.graph.as_default()
    tf.import_graph_def(graph_def, name='')
    for i, n in enumerate(graph_def.node):
        print("Name of the node - %s" % n.name)

    image = cv.imdecode(np.fromfile('testImg/Part3.jpg', dtype=np.uint8), 1)
    # Convert to single channel
    gray = cv.cvtColor(image, cv.COLOR_RGB2GRAY)  
    #gray.resize(gray, (512, 1280), cv.INTER_LINEAR) 
    #height = gray.shape[0]   
    #width = gray.shape[1]    
    cv.imwrite('gray.jpg', gray)

    # input
    input_x = sess.graph.get_tensor_by_name('Image:0')
    #input_seq_len = sess.graph.get_tensor_by_name('seq_len:0')
    # output
    output = sess.graph.get_tensor_by_name('segment/Sigmoid:0')

    # Run the model
    #out = sess.run([output, input_x], feed_dict={'Image:0': [np.asarray(resized_image)]})
    # Due to the single channel used for Image training, the following channel should be changed to 1
    mask_batch ,output_batch = sess.run([output, input_x], feed_dict={'Image:0': gray.reshape(1, 1280, 512, 1)})
    #print(mask_batch[0])
    #print(len(mask_batch))
    #print(type(mask_batch))
    mask=np.array(mask_batch[0]).squeeze(2)*255  # mask is white
    # The first way to save results
    img_visual=concatImage([mask])
    img_visual.save('PIL_mask.jpg')
    # The second way to save results
    cv.imwrite('CV_mask.jpg', mask)
    #Note: the length and width of the saved mask are 0.4 times that of the original figure (1280,512)
    