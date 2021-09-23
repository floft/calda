"""
Datasets
"""
import os
import math
import tensorflow as tf

from absl import app
from absl import flags

from datasets import datasets
from datasets.tfrecord import tfrecord_filename

FLAGS = flags.FLAGS

flags.DEFINE_integer("train_batch", 128, "Batch size for training")
flags.DEFINE_integer("eval_batch", 2048, "Batch size for evaluation")
flags.DEFINE_enum("batch_division", "all", ["none", "sources", "all"], "Batch size options (e.g. 32): none - 32 for each source and target; sources - 32/n for n sources, 32 for target; all - 32/(n+1) for n sources and 1 target. Overridden as \"sources\" for generalization and weak supervision methods.")
flags.DEFINE_integer("shuffle_buffer", 60000, "Dataset shuffle buffer size")
flags.DEFINE_integer("prefetch_buffer", 1, "Dataset prefetch buffer size (0 = autotune)")
flags.DEFINE_boolean("tune_num_parallel_calls", False, "Autotune num_parallel_calls")
flags.DEFINE_integer("eval_shuffle_seed", 0, "Evaluation shuffle seed for repeatability")
flags.DEFINE_boolean("max_examples_percentage", True, "Make {train,eval}_max_examples a percentage of total rather than number of examples")
flags.DEFINE_float("train_max_examples", 0, "Max number of examples to use for training (default 0, i.e. all). If max_examples_percentage=True, this is in [0,1] and a percentage.")
flags.DEFINE_float("max_target_examples", 0, "Max number of target examples to use during training (default 0, i.e. all; overrides train_max_examples for target). If max_examples_percentage=True, this is in [0,1] and a percentage.")
flags.DEFINE_float("eval_max_examples", 0, "Max number of examples to evaluate for validation (default 0, i.e. all). If max_examples_percentage=True, this is in [0,1] and a percentage.")
flags.DEFINE_integer("trim_time_steps", 0, "For testing RNN vs. CNN handling varying time series length, allow triming to set size (default 0, i.e. use all data)")
flags.DEFINE_integer("trim_features", 0, "For testing RNN vs. CNN handling varying numbers of features, allow only using the first n features (default 0, i.e. all the features)")
flags.DEFINE_string("source_feature_subset", "", "Comma-separated zero-indexed integer list of which features (and in which order) to use for the source domain (default blank, i.e. all the features)")
flags.DEFINE_string("target_feature_subset", "", "Comma-separated zero-indexed integer list of which features (and in which order) to use for the target domain (default blank, i.e. all the features)")
flags.DEFINE_boolean("cache", True, "Cache datasets in memory to reduce filesystem usage")


class Dataset:
    """ Load datasets from tfrecord files """
    def __init__(self, num_classes, class_labels, num_domains, num_modalities,
            train_filenames, test_filenames, domain_id=None,
            train_batch=None, eval_batch=None,
            shuffle_buffer=None, prefetch_buffer=None,
            eval_shuffle_seed=None, cache=None,
            train_max_examples=None, eval_max_examples=None,
            tune_num_parallel_calls=None, feature_subset=None,
            modality_subset=None, max_examples_percentage=None):
        """
        Initialize dataset

        Must specify num_classes and class_labels (the names of the classes).
        Other arguments if None are defaults from command line flags.

        For example:
            Dataset(num_classes=2, class_labels=["class1", "class2"])
        """
        # Sanity checks
        assert num_classes == len(class_labels), \
            "num_classes != len(class_labels)"

        # Set parameters
        self.num_classes = num_classes
        self.class_labels = class_labels
        self.num_domains = num_domains
        self.num_modalities = num_modalities
        self.domain_id = domain_id
        self.train_batch = train_batch
        self.eval_batch = eval_batch
        self.shuffle_buffer = shuffle_buffer
        self.prefetch_buffer = prefetch_buffer
        self.eval_shuffle_seed = eval_shuffle_seed
        self.cache = cache
        self.eval_max_examples = eval_max_examples
        self.train_max_examples = train_max_examples
        self.tune_num_parallel_calls = tune_num_parallel_calls
        self.feature_subset = feature_subset
        self.modality_subset = modality_subset
        self.max_examples_percentage = max_examples_percentage

        # Unaffected by the modality subset since we need to know how many
        # modalities are in the tfrecord file when loading
        self.tfrecord_num_modalities = num_modalities

        # If we set a modality subset, then make sure they exist and then update
        # num_modalities to the subset we chose
        if modality_subset is not None:
            for modality in modality_subset:
                assert modality < num_modalities, \
                    str(modality) + " is not < num_modalities = " \
                    + str(num_modalities)

            self.num_modalities = len(modality_subset)

        # Set defaults if not specified
        if self.train_batch is None:
            self.train_batch = FLAGS.train_batch
        if self.eval_batch is None:
            self.eval_batch = FLAGS.eval_batch
        if self.shuffle_buffer is None:
            self.shuffle_buffer = FLAGS.shuffle_buffer
        if self.prefetch_buffer is None:
            self.prefetch_buffer = FLAGS.prefetch_buffer
        if self.eval_shuffle_seed is None:
            self.eval_shuffle_seed = FLAGS.eval_shuffle_seed
        if self.cache is None:
            self.cache = FLAGS.cache
        if self.eval_max_examples is None:
            self.eval_max_examples = FLAGS.eval_max_examples
        if self.train_max_examples is None:
            self.train_max_examples = FLAGS.train_max_examples
        if self.tune_num_parallel_calls is None:
            self.tune_num_parallel_calls = FLAGS.tune_num_parallel_calls
        if self.max_examples_percentage is None:
            self.max_examples_percentage = FLAGS.max_examples_percentage

        # Load the dataset
        self.train, self.train_evaluation, self.test_evaluation = \
            self.load_dataset(train_filenames, test_filenames)

    def get_feature_description(self):
        """
        Create feature description based on the number of modalities

        For example, if self.tfrecord_num_modalities = 2, we get:
            x_0, x_1, y
        """
        modalities = []

        for i in range(self.tfrecord_num_modalities):
            modalities.append("x_"+str(i))

        # All datasets have a label (y) "feature" and example id (id)
        modalities += ["y", "id"]

        # Create a description of the features
        # See: https://www.tensorflow.org/tutorials/load_data/tf-records
        def _feature():
            return tf.io.FixedLenFeature([], tf.string)

        feature_description = {}

        for modality in modalities:
            feature_description[modality] = _feature()

        return feature_description

    def load_tfrecords(self, filenames, batch_size, count=False, evaluation=False):
        """
        Load data from .tfrecord files (requires less memory but more disk space)
        """
        if len(filenames) == 0:
            return None

        # Compute feature description based on the number of modalities
        feature_description = self.get_feature_description()

        def _parse_example_function(example_proto):
            """
            Parse the input tf.Example proto using the dictionary above.
            parse_single_example is without a batch, parse_example is with batches

            What's parsed returns byte strings, but really we want to get the
            tensors back that we encoded with tf.io.serialize_tensor() earlier,
            so also run tf.io.parse_tensor
            """
            parsed = tf.io.parse_single_example(serialized=example_proto,
                features=feature_description)

            # Sort names for consistency
            names = list(parsed.keys())
            names.sort()

            xs = []

            for name in names:
                # Skip "y" and "id" since we know these are here
                if name == "y" or name == "id":
                    continue

                # Parse
                x = tf.io.parse_tensor(parsed[name], tf.float32)

                # Trim to certain time series length (note single example, not batch)
                # shape before: [time_steps, features]
                # shape after:  [min(time_steps, trim_time_steps), features]
                if FLAGS.trim_time_steps != 0:
                    x = tf.slice(x, [0, 0],
                        [tf.minimum(tf.shape(x)[0], FLAGS.trim_time_steps), tf.shape(x)[1]])

                # Trim to a certain number of features (the first n = trim_features)
                if FLAGS.trim_features != 0:
                    x = tf.slice(x, [0, 0],
                        [tf.shape(x)[0], tf.minimum(tf.shape(x)[1], FLAGS.trim_features)])

                # Select only the desired features, if specified
                if self.feature_subset is not None:
                    assert FLAGS.trim_features == 0, \
                        "cannot specify both {source,target}_feature_subset and trim_features"
                    # axis=-1 is the feature dimension
                    x = tf.gather(x, self.feature_subset, axis=-1)

                xs.append(x)

            # Rearrange or take subset if desired
            if self.modality_subset is not None:
                xs_subset = []

                for modality in self.modality_subset:
                    xs_subset.append(xs[modality])

                xs = xs_subset

            # Label (y)
            y = tf.io.parse_tensor(parsed["y"], tf.float32)

            # Example ID - unique to domain/set (e.g. within source 1 train)
            example_id = tf.io.parse_tensor(parsed["id"], tf.int32)

            # TensorFlow gives weird problems when lists, probably wants to
            # convert lists to tensors, which means all the x's have to be the
            # same shape, which defeats the purpose of separating them here.
            return tuple(xs), y, example_id

        # Interleave the tfrecord files
        files = tf.data.Dataset.from_tensor_slices(filenames)
        dataset = files.interleave(
            lambda x: tf.data.TFRecordDataset(x, compression_type='GZIP').prefetch(100),
            cycle_length=len(filenames), block_length=1)

        # If desired, take the first max_examples examples or percentage.
        #
        # Note: this is the first so-many examples, but we shuffled before
        # putting into the tfrecord file (train_test_split in datasets.py), so
        # it is a random set essentially. We do this so we consistently use the
        # same data between runs.
        examples = 0

        if self.max_examples_percentage:
            # Take the first max_examples percentage of the total examples
            if evaluation:
                percentage = self.eval_max_examples
            else:
                percentage = self.train_max_examples

            # To save time, if the percentage is zero, just skip all of this
            # and set examples=0, i.e. use them all
            if percentage > 0:
                # Calculate total number of examples
                total = 0

                for x in dataset:
                    total += 1

                # Take percentage -- ceil so we'd end up with 1 not 0 if we
                # have very few examples
                examples = math.ceil(percentage * total)

                print(percentage, "percent of", total, "examples =", examples, "for", filenames)
            else:
                examples = 0
        else:
            # Take the first max_examples examples
            if evaluation:
                examples = int(self.eval_max_examples)
            else:
                examples = int(self.train_max_examples)

        if examples > 0:
            dataset = dataset.take(examples)

        # Whether to do autotuning of prefetch or num_parallel_calls
        prefetch_buffer = self.prefetch_buffer
        num_parallel_calls = None
        if self.tune_num_parallel_calls:
            num_parallel_calls = tf.data.experimental.AUTOTUNE
        if self.prefetch_buffer == 0:
            prefetch_buffer = tf.data.experimental.AUTOTUNE

        # Use .cache() or .cache(filename) to reduce loading over the network
        # https://www.tensorflow.org/guide/data_performance#map_and_cache
        # Example: https://www.tensorflow.org/tutorials/load_data/images
        if self.cache:
            # Map before caching so we don't have to keep doing this over and over
            # again -- drastically reduces CPU usage.
            dataset = dataset.map(_parse_example_function,
                num_parallel_calls=num_parallel_calls)

            dataset = dataset.cache()

        if count:  # only count, so no need to shuffle
            pass
        elif evaluation:  # don't repeat since we want to evaluate entire set
            dataset = dataset.shuffle(self.shuffle_buffer, seed=self.eval_shuffle_seed)
            pass
        else:  # repeat and shuffle
            dataset = dataset.shuffle(self.shuffle_buffer).repeat()

        # If not caching, then it's faster to map right next to batch
        if not self.cache:
            dataset = dataset.map(_parse_example_function,
                num_parallel_calls=num_parallel_calls)

        dataset = dataset.batch(batch_size)
        dataset = dataset.prefetch(prefetch_buffer)

        return dataset

    def load_dataset(self, train_filenames, test_filenames):
        """
        Load the X dataset as a tf.data.Dataset from train/test tfrecord filenames
        """
        train_dataset = self.load_tfrecords(
            train_filenames, self.train_batch)
        eval_train_dataset = self.load_tfrecords(
            train_filenames, self.eval_batch, evaluation=True)
        eval_test_dataset = self.load_tfrecords(
            test_filenames, self.eval_batch, evaluation=True)

        return train_dataset, eval_train_dataset, eval_test_dataset

    def label_to_int(self, label_name):
        """ e.g. Bathe to 0 """
        return self.class_labels.index(label_name)

    def int_to_label(self, label_index):
        """ e.g. 0 to Bathe """
        return self.class_labels[label_index]


def load(dataset_name, num_domains, postfix="", test=False,
        train_on_everything=False, *args, **kwargs):
    """ Load a dataset (source and target). Names must be in datasets.names().

    If test=True, then load real test set. Otherwise, load validation set as
    the "test" data (for use during training and hyperparameter tuning).
    """
    # Sanity checks
    assert dataset_name in names(), \
        dataset_name + " not a supported dataset"

    # Get dataset information
    num_classes, class_labels = datasets.attributes(dataset_name)

    # Get dataset tfrecord filenames
    def _path(filename):
        """ Files are in datasets/ subdirectory. If the file exists, return it
        as an array since we may sometimes want more than one file for a
        dataset. If it doesn't exist, ignore it (some datasets don't have a test
        set for example)."""
        fn = os.path.join("datasets", "tfrecords", filename)
        return [fn] if os.path.exists(fn) else []

    if postfix != "":
        postfix = postfix + "_"

    train_filenames = _path(tfrecord_filename(dataset_name, postfix+"train"))
    valid_filenames = _path(tfrecord_filename(dataset_name, postfix+"valid"))
    test_filenames = _path(tfrecord_filename(dataset_name, postfix+"test"))

    # This is used for some plots not actual training
    if train_on_everything:
        print("Warning: training dataset contains all train/valid/test data")
        train_filenames += valid_filenames + test_filenames
        test_filenames = []
    # By default use validation data as the "test" data, unless test=True
    elif not test:
        test_filenames = valid_filenames
    # If test=True, then make "train" consist of both training and validation
    # data to match the original dataset.
    else:
        train_filenames += valid_filenames

    # Get number of modalities for this dataset
    num_modalities = get_num_modalities(dataset_name)

    # Create all the train, test, evaluation, ... tf.data.Dataset objects within
    # a Dataset() class that stores them
    dataset = Dataset(num_classes, class_labels, num_domains, num_modalities,
        train_filenames, test_filenames, *args, **kwargs)

    return dataset


def load_da(dataset, sources, target, *args,
        source_modality_subset="", target_modality_subset="",
        override_batch_division=False, **kwargs):
    """
    Load the source(s) and target domains

    Input:
        dataset - one of the dataset names (e.g. ucihar)
        sources - comma-separated string of source domain numbers
        target - string of target domain number

    Returns:
        [source1_dataset, source2_dataset, ...], target_dataset
    """
    # Allow target blank meaning None as well
    if target == "":
        target = None

    # Get proper dataset names
    source_ids = sources.split(",")
    sources = [dataset+"_"+x for x in source_ids]

    # Need to know how many domains for creating the proper-sized model, etc.
    num_domains = len(sources)

    if target is not None:
        # Probably 1, but still will work if we ever support multiple targets
        target_ids = target.split(",")
        num_domains += len(target_ids)

        original_target = target
        target = dataset+"_"+target

    # Integer ids
    if target is not None:
        assert len(target_ids) == 1, "only support one target domain for now"
        target_id = int(target_ids[0])
    else:
        target_id = None

    source_ids = [int(source_id) for source_id in source_ids]

    # Check they're all valid
    valid_names = names()

    for s in sources:
        assert s in valid_names, "unknown source domain: " + s \
            + " not in " + str(valid_names)

    if target is not None:
        assert target in valid_names, "unknown target domain: "+target

    # Determine batch sizes
    source_train_batch = None
    target_train_batch = None

    # Divide among sources, so batch_size/num_sources. Keep target the normal
    # batch size. Though, we must at least have one sample from each domain,
    # so take max of the division and 1.
    #
    # Note: we don't need to change eval_batch since that data is fed in
    # per-domain anyway in metrics.py. Thus, there's no point in decreasing
    # the batch size since it's not affected by the number of domains.
    assert FLAGS.train_batch > 0, "must have positive train_batch size"
    if FLAGS.batch_division == "sources" or override_batch_division:
        source_train_batch = max(FLAGS.train_batch // len(sources), 1)
        target_train_batch = FLAGS.train_batch

        if override_batch_division:
            print("Overriding: batch_division=sources")
    # Divide among all, so batch_size/num_domains. Set for both sources/target.
    elif FLAGS.batch_division == "all":
        batch_size = max(FLAGS.train_batch // num_domains, 1)
        source_train_batch = batch_size
        target_train_batch = batch_size
    else:
        source_train_batch = FLAGS.train_batch
        target_train_batch = FLAGS.train_batch

    #print("Source batch size:", source_train_batch, "for", len(sources), "sources")
    #print("Target batch size:", target_train_batch)

    # Which features from source/target to use (if not all of them)
    if FLAGS.source_feature_subset == "":
        source_feature_subset = None
    else:
        source_feature_subset = [int(x) for x in FLAGS.source_feature_subset.split(",")]

    if FLAGS.target_feature_subset == "":
        target_feature_subset = None
    else:
        target_feature_subset = [int(x) for x in FLAGS.target_feature_subset.split(",")]

    # Which modalities from source/target to use (if not all of them)
    if source_modality_subset == "":
        source_modality_subset = None
    else:
        source_modality_subset = [int(x) for x in source_modality_subset.split(",")]

    if target_modality_subset == "":
        target_modality_subset = None
    else:
        target_modality_subset = [int(x) for x in target_modality_subset.split(",")]

    # Load each source
    source_datasets = []

    for i, s in enumerate(sources):
        source_datasets.append(load(s, num_domains, *args,
            train_batch=source_train_batch,
            feature_subset=source_feature_subset,
            modality_subset=source_modality_subset,
            domain_id=source_ids[i],
            **kwargs))

    # Check that they all have the same number of classes as the first one
    for i in range(1, len(source_datasets)):
        assert source_datasets[i].num_classes == source_datasets[0].num_classes, \
            "Source domain "+str(i)+" has different # of classes than source 0"

    # Load target
    if target is not None:
        train_max_examples = None
        eval_max_examples = None

        # If desired, only use the a limited number of target examples for
        # training/evaluation
        if FLAGS.max_target_examples != 0:
            train_max_examples = FLAGS.max_target_examples

            # Note: for now don't limit eval examples. We want the best estimate
            # of evaluation performance. We just want to limit how much data
            # we have during training.
            #eval_max_examples = FLAGS.max_target_examples

        target_dataset = load(target, num_domains, *args,
            train_batch=target_train_batch,
            train_max_examples=train_max_examples,
            eval_max_examples=eval_max_examples,
            feature_subset=target_feature_subset,
            modality_subset=target_modality_subset,
            domain_id=target_id,
            **kwargs)

        # Check that the target has the same number of classes as the first
        # source (since we already verified all sources have the same)
        assert target_dataset.num_classes == source_datasets[0].num_classes, \
            "Target has different # of classes than source 0"

        # Check that the target is in the allowed set of target users
        allowed_target_users = datasets.get_dataset_target_users(dataset)
        assert int(original_target) in allowed_target_users, \
            "target {} not in list of allowed target users {}".format(
                original_target, allowed_target_users)
    else:
        target_dataset = None

    return source_datasets, target_dataset


def names():
    """ Returns list of all the available datasets to load """
    return datasets.names()


def get_num_modalities(dataset_name):
    """ Get the number of modalities for each dataset """
    # Different datasets have different sets of features
    # Format for the names is: watch_1 where "watch" is the dataset_name
    # and 1 is the user_id
    #
    # Note: this allows for the dataset to have _ in it by recombining the
    # multiple splits and ignoring the last one that's the user id.
    parts = dataset_name.split("_")
    assert len(parts) >= 2, "cannot split dataset into name_user"
    dataset_name = "_".join(parts[:-1])  # re-combine _-separated name
    # user_id = parts[-1]  # just the final user id

    num_modalities = datasets.get_dataset_modalities(dataset_name)

    return num_modalities


def main(argv):
    print("Available datasets:", names())

    # Example showing that the sizes and number of channels are matched
    sources, target = load_da("ucihar", "1,2", "3")

    print("Source 0:", sources[0].train)
    print("Target:", target.train)

    for i, source in enumerate(sources):
        assert source.train is not None, "dataset file probably doesn't exist"

        for x, y in source.train:
            print("Source "+str(i)+" x shape:", x.shape)
            print("Source "+str(i)+" y shape:", y.shape)
            break

    assert target.train is not None, "dataset file probably doesn't exist"

    for x, y in target.train:
        print("Target x shape:", x.shape)
        print("Target y shape:", y.shape)
        break


if __name__ == "__main__":
    app.run(main)
