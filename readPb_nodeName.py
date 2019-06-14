import tensorflow as tf

# Read the graph.
with tf.gfile.FastGFile('./checkpoint/frozen_model.pb', 'rb') as f:
    graph_def = tf.GraphDef()
    graph_def.ParseFromString(f.read())

config = tf.ConfigProto(allow_soft_placement=True,log_device_placement=False)
config.gpu_options.allow_growth = True


with tf.Session(config=config) as sess:
    # Restore session
    sess.graph.as_default()
    tf.import_graph_def(graph_def, name='')
    for i, n in enumerate(graph_def.node):
        print("Name of the node - %s" % n.name)

