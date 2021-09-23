"""
Functions to write the x,y data to a tfrecord file
"""
import tensorflow as tf


def _bytes_feature(value):
    """ Returns a bytes_list from a string / byte. """
    if isinstance(value, type(tf.constant(0))):
        value = value.numpy()  # BytesList won't unpack a string from an EagerTensor.
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))


def create_tf_example(x, y, example_id):
    tf_example = tf.train.Example(features=tf.train.Features(feature={
        'x': _bytes_feature(tf.io.serialize_tensor(x)),
        'y': _bytes_feature(tf.io.serialize_tensor(y)),
        'id': _bytes_feature(tf.io.serialize_tensor(example_id)),
    }))
    return tf_example


def create_tf_example_modality(xs, y, example_id):
    """
    Create TF example based on the number of modalities

    For example, if self.num_modalities = 2, we get features:
        x_0, x_1, y
    """
    # Xs
    num_modalities = len(xs)
    modalities = []

    for i in range(num_modalities):
        modalities.append("x_"+str(i))

    def _feature(x):
        return _bytes_feature(tf.io.serialize_tensor(x))

    feature = {}

    for i, modality in enumerate(modalities):
        feature[modality] = _feature(xs[i])

    # y
    feature["y"] = _feature(y)

    # example id
    feature["id"] = _feature(example_id)

    # example containing both xs and y
    tf_example = tf.train.Example(features=tf.train.Features(
        feature=feature))

    return tf_example


class MultiModalityTFRecord:
    """
    Version used in the multimodal/ code, writing each example individually
    """
    def __init__(self, filename):
        options = tf.io.TFRecordOptions(compression_type="GZIP")
        self.writer = tf.io.TFRecordWriter(filename, options=options)
        self.example_id = 0

    def write(self, xs, epochs):
        # To match the tfrecord format of the non-multi-modal datasets, split
        # the last xs (the response) "modality" into a separate y. We also
        # throw out the epochs at the moment since we don't use them.
        y = xs[-1]
        xs = xs[:-1]

        # If we don't "squeeze" y, then we end up with (n,1) when loading y
        # from the tfrecords. We instead want (n,). Though, each individual
        # y changes from being shape (1,) to ().
        assert len(y.shape) == 1, \
            "y.shape not of length 1 but " + str(y.shape)
        assert y.shape[0] == 1, \
            "expecting y to be shape (1,) for each examples but is " + str(y.shape)
        y = tf.squeeze(y)

        tf_example = create_tf_example_modality(xs, y, self.example_id)
        self.writer.write(tf_example.SerializeToString())
        self.example_id += 1

    def close(self):
        # Normally the tf.io.TFRecordWriter is used in a with block
        self.writer.close()


def write_tfrecord(filename, x, y):
    """ Output to TF record file """
    assert len(x) == len(y)
    options = tf.io.TFRecordOptions(compression_type="GZIP")

    with tf.io.TFRecordWriter(filename, options=options) as writer:
        for i in range(len(x)):
            tf_example = create_tf_example(x[i], y[i], i)
            writer.write(tf_example.SerializeToString())


def write_tfrecord_modality(filename, xs, y):
    """ Output to TF record file, multi-modality version

    xs = [(example 1 modality 1, example 2 modality 1, ...),
          (example 1 modality 2, example 2 modality 2, ...)]

    Note: write_tfrecord_modality_by_example() has a different format for x.
    This version is more convenient to create, but basically converts to the
    format that write_tfrecord_modality_by_example() uses since that is the
    format needed to write it to disk.
    """
    num_modalities = len(xs)
    for x in xs:
        assert len(x) == len(y), "each modality must have same length as y"
    options = tf.io.TFRecordOptions(compression_type="GZIP")

    with tf.io.TFRecordWriter(filename, options=options) as writer:
        for i in range(len(y)):
            # Get the x for each of this example's modality along with the
            # corresponding y value
            tf_example = create_tf_example_modality(
                [xs[m][i] for m in range(num_modalities)], y[i], i)
            writer.write(tf_example.SerializeToString())


def write_tfrecord_modality_by_example(filename, x, y):
    """ Output to TF record file -- exactly the same as write_tfrecord except
    this calls the multi-modality version of create_tf_example

    x = [(example 1 modality 1, example 1 modality 2, ...), ...]
    """
    assert len(x) == len(y)
    options = tf.io.TFRecordOptions(compression_type="GZIP")

    with tf.io.TFRecordWriter(filename, options=options) as writer:
        for i in range(len(x)):
            tf_example = create_tf_example_modality(x[i], y[i], i)
            writer.write(tf_example.SerializeToString())


def tfrecord_filename(dataset_name, postfix):
    """ Filename for tfrecord files, e.g. ucihar_1_train.tfrecord """
    return "%s_%s.tfrecord"%(dataset_name, postfix)


def tfrecord_as_images_folder(dataset_name, postfix):
    """ Filename for tfrecord files, e.g. ucihar_1_train/ """
    return "%s_%s"%(dataset_name, postfix)
