"""
Single-modality datasets (see multimodal/ for multi-modality datasets)

Load the desired datasets into memory so we can write them to tfrecord files
in generate_tfrecords.py
"""
import os
import zipfile
import tarfile
import scipy.io
import numpy as np
import tensorflow as tf

from sklearn.model_selection import train_test_split

from absl import app
from absl import flags

try:
    import datasets.MyoArmbandDataset.PyTorchImplementation.RawEnhancedConvNet.load_evaluation_dataset as load_myo_data
except ImportError:
    print("Warning: could not import Myo dataset loading, will only work from tfrecord files")

FLAGS = flags.FLAGS

flags.DEFINE_enum("normalize", "meanstd", ["none", "minmax", "meanstd"], "How to normalize data")

list_of_datasets = {}


def register_dataset(name):
    """ Add dataset to the list of datsets, e.g. add @register_dataset("name")
    before a class definition """
    assert name not in list_of_datasets, "duplicate dataset named " + name

    def decorator(cls):
        list_of_datasets[name] = cls
        return cls

    return decorator


def get_dataset(name):
    """ Based on the given name, get the correct dataset processor

    Note: if not in this file of single-modality datasets, try the multi-modal
    datasets to see if it's there. If not, error.
    """
    # Check single-modality datasets
    if name in list_of_datasets.keys():
        return list_of_datasets[name]

    # Check multi-modality datasets
    name = name.replace("mm_", "")  # denotes multi-modal
    raise NotImplementedError("Unknown dataset name " + name
        + ", not in "+str(list_datasets()))


def get_dataset_users(name):
    """ Get list of users for a dataset """
    return get_dataset(name).users


def get_dataset_target_users(name):
    """ Get list of target users for a dataset, if target_users doesn't exist
    in the class, then it's the same as get_dataset_users() """
    d = get_dataset(name)

    if hasattr(d, "target_users") and d.target_users is not None:
        return d.target_users

    return d.users


def get_dataset_modalities(name):
    """ Get number of modalities for a dataset """
    return get_dataset(name).num_modalities


def call_dataset(name, *args, **kwargs):
    """ Based on the given name, call the correct dataset processor """
    return get_dataset(name)(*args, **kwargs)


def list_datasets():
    """ Returns list of all the available datasets -- both single-modality
    and multi-modality"""
    return list(list_of_datasets.keys())


def list_datasets_single_modality():
    """ Returns list of all the available datasets (only single-modality) """
    return list(list_of_datasets.keys())


def zero_to_n(n):
    """ Return [0, 1, 2, ..., n] """
    return list(range(0, n+1))


def one_to_n(n):
    """ Return [1, 2, 3, ..., n] """
    return list(range(1, n+1))


class Dataset:
    """
    Base class for datasets

    class Something(Dataset):
        num_classes = 2
        class_labels = ["class1", "class2"]
        num_modalities = 1
        window_size = 250
        window_overlap = False

        def __init__(self, *args, **kwargs):
            super().__init__(Something.num_classes, Something.class_labels,
                Something.num_modalities,
                Something.window_size, Something.window_overlap,
                *args, **kwargs)

        def process(self, data, labels):
            ...
            return super().process(data, labels)

        def load(self):
            ...
            return train_data, train_labels, test_data, test_labels

    Also, add to the datasets={"something": Something, ...} dictionary below.
    """
    already_normalized = False

    def __init__(self, num_classes, class_labels, num_modalities,
            window_size, window_overlap,
            feature_names=None, test_percent=0.2):
        """
        Initialize dataset

        Must specify num_classes and class_labels (the names of the classes).

        For example,
            Dataset(num_classes=2, class_labels=["class1", "class2"])

        This calls load() to get the data, process() to normalize, convert to
        float, etc.

        At the end, look at dataset.{train,test}_{data,labels}
        """
        # Sanity checks
        assert num_classes == len(class_labels), \
            "num_classes != len(class_labels)"

        # Set parameters
        self.num_classes = num_classes
        self.class_labels = class_labels
        self.num_modalities = num_modalities
        self.window_size = window_size
        self.window_overlap = window_overlap
        self.feature_names = feature_names
        self.test_percent = test_percent

        assert len(self.feature_names) == self.num_modalities, \
            "feature_names should be same length as num_modalities"

        # Load the dataset
        train_data, train_labels, test_data, test_labels = self.load()

        if train_data is not None and train_labels is not None:
            self.train_data, self.train_labels = \
                self.process(train_data, train_labels)
        else:
            self.train_data = None
            self.train_labels = None

        if test_data is not None and test_labels is not None:
            self.test_data, self.test_labels = \
                self.process(test_data, test_labels)
        else:
            self.test_data = None
            self.test_labels = None

    def load(self):
        raise NotImplementedError("must implement load() for Dataset class")

    def download_dataset(self, files_to_download, url):
        """
        Download url/file for file in files_to_download
        Returns: the downloaded filenames for each of the files given

        Note: files_to_download can either be a list of individual filenames
        or if desired a list of tuples:
            [(filename used when saving, path to download), ...]
        """
        downloaded_files = []

        for f in files_to_download:
            if isinstance(f, tuple):
                save_f, f = f
            else:
                save_f = f

            downloaded_files.append(tf.keras.utils.get_file(
                fname=save_f, origin=url+"/"+f))

        return downloaded_files

    def process(self, data, labels):
        """ Perform conversions, etc. If you override,
        you should `return super().process(data, labels)` to make sure these
        options are handled. """
        return data, labels

    def train_test_split(self, x, y, random_state=42):
        """
        Split x and y data into train/test sets

        Warning: train_test_split() is from sklearn but self.train_test_split()
        is this function, which is what you should use.
        """
        x_train, x_test, y_train, y_test = \
            train_test_split(x, y, test_size=self.test_percent,
            stratify=y, random_state=random_state)
        return x_train, y_train, x_test, y_test

    def train_test_split_modality(self, xs, y, random_state=42):
        """
        Split x and y data into train/test sets. Assumes xs is a list of the
        modalities in the format

        xs = [(example 1 modality 1, example 2 modality 1, ...),
              (example 1 modality 2, example 2 modality 2, ...)]
        """
        # Returns train/test for each input passed in, so: xs1_train, xs1_test,
        # xs2_train, xs2_test, ...
        results = train_test_split(*xs, y, test_size=self.test_percent,
            stratify=y, random_state=random_state)

        # Train is evens (starting at position 0), test is odds
        assert len(results) % 2 == 0, "should get even number of splits"
        train = results[0::2]
        test = results[1::2]

        # y is at the end, xs is everything else
        xs_train = train[:-1]
        xs_test = test[:-1]
        y_train = train[-1]
        y_test = test[-1]

        return xs_train, y_train, xs_test, y_test

    def get_file_in_archive(self, archive, filename):
        """ Read one file out of the already-open zip/rar file """
        with archive.open(filename) as fp:
            contents = fp.read()
        return contents

    def window_next_i(self, i, overlap, window_size):
        """ Where to start the next window """
        if overlap is not False:
            if overlap is True:
                i += 1
            elif isinstance(overlap, int):
                i += overlap
            else:
                raise NotImplementedError("overlap should be True/False or integer")
        else:
            i += window_size

        return i

    def create_windows_x(self, x, window_size, overlap):
        """
        Concatenate along dim-1 to meet the desired window_size. We'll skip any
        windows that reach beyond the end. Only process x (saves memory).

        Three options (examples for window_size=5):
            Overlap - e.g. window 0 will be a list of examples 0,1,2,3,4 and the
                label of example 4; and window 1 will be 1,2,3,4,5 and the label of
                example 5
            No overlap - e.g. window 0 will be a list of examples 0,1,2,3,4 and the
                label of example 4; and window 1 will be 5,6,7,8,9 and the label of
                example 9
            Overlap as integer rather than True/False - e.g. if overlap=2 then
                window 0 will be examples 0,1,2,3,4 and then window 1 will be
                2,3,4,5,6, etc.
        """
        x = np.expand_dims(x, axis=1)

        # No work required if the window size is 1, only part required is
        # the above expand dims
        if window_size == 1:
            return x

        windows_x = []
        i = 0

        while i < len(x)-window_size:
            window_x = np.expand_dims(np.concatenate(x[i:i+window_size], axis=0), axis=0)
            windows_x.append(window_x)
            # Where to start the next window
            i = self.window_next_i(i, overlap, window_size)

        return np.vstack(windows_x)

    def create_windows_y(self, y, window_size, overlap):
        """
        Concatenate along dim-1 to meet the desired window_size. We'll skip any
        windows that reach beyond the end. Only process y (saves memory).

        Two options (examples for window_size=5):
            Overlap - e.g. window 0 will be a list of examples 0,1,2,3,4 and the
                label of example 4; and window 1 will be 1,2,3,4,5 and the label of
                example 5
            No overlap - e.g. window 0 will be a list of examples 0,1,2,3,4 and the
                label of example 4; and window 1 will be 5,6,7,8,9 and the label of
                example 9
            Overlap as integer rather than True/False - e.g. if overlap=2 then
                window 0 will be examples 0,1,2,3,4 and then window 1 will be
                2,3,4,5,6, etc.
        """
        # No work required if the window size is 1
        if window_size == 1:
            return y

        windows_y = []
        i = 0

        while i < len(y)-window_size:
            window_y = y[i+window_size-1]
            windows_y.append(window_y)
            # Where to start the next window
            i = self.window_next_i(i, overlap, window_size)

        return np.hstack(windows_y)

    def create_windows(self, x, y, window_size, overlap):
        """ Split time-series data into windows """
        x = self.create_windows_x(x, window_size, overlap)
        y = self.create_windows_y(y, window_size, overlap)
        return x, y

    def pad_to(self, data, desired_length):
        """
        Pad the number of time steps to the desired length

        Accepts data in one of two formats:
            - shape: (time_steps, features) -> (desired_length, features)
            - shape: (batch_size, time_steps, features) ->
                (batch_size, desired_length, features)
        """
        if len(data.shape) == 2:
            current_length = data.shape[0]
            assert current_length <= desired_length, \
                "Cannot shrink size by padding, current length " \
                + str(current_length) + " vs. desired_length " \
                + str(desired_length)
            return np.pad(data, [(0, desired_length - current_length), (0, 0)],
                    mode="constant", constant_values=0)
        elif len(data.shape) == 3:
            current_length = data.shape[1]
            assert current_length <= desired_length, \
                "Cannot shrink size by padding, current length " \
                + str(current_length) + " vs. desired_length " \
                + str(desired_length)
            return np.pad(data, [(0, 0), (0, desired_length - current_length), (0, 0)],
                    mode="constant", constant_values=0)
        else:
            raise NotImplementedError("pad_to requires 2 or 3-dim data")

    def label_to_int(self, label_name):
        """ e.g. Bathe to 0 """
        return self.class_labels.index(label_name)

    def int_to_label(self, label_index):
        """ e.g. Bathe to 0 """
        return self.class_labels[label_index]


@register_dataset("ucihar")
class UciHarBase(Dataset):
    """
    Loads human activity recognition data files in datasets/UCI HAR Dataset.zip

    Download from:
    https://archive.ics.uci.edu/ml/datasets/Human+Activity+Recognition+Using+Smartphones
    """
    feature_names = [[
        "body_acc_x", "body_acc_y", "body_acc_z",
        "body_gyro_x", "body_gyro_y", "body_gyro_z",
        "total_acc_x", "total_acc_y", "total_acc_z",
    ]]
    num_classes = 6
    class_labels = [
        "walking", "walking_upstairs", "walking_downstairs",
        "sitting", "standing", "laying",
    ]
    num_modalities = 1
    users = one_to_n(30)  # 30 people
    already_normalized = True

    def __init__(self, users, *args, **kwargs):
        self.users = users
        super().__init__(UciHarBase.num_classes, UciHarBase.class_labels,
            UciHarBase.num_modalities,
            None, None, UciHarBase.feature_names, *args, **kwargs)

    def download(self):
        (dataset_fp,) = self.download_dataset(["UCI%20HAR%20Dataset.zip"],
            "https://archive.ics.uci.edu/ml/machine-learning-databases/00240")
        return dataset_fp

    def get_feature(self, content):
        """
        Read the space-separated, example on each line file

        Returns 2D array with dimensions: [num_examples, num_time_steps]
        """
        lines = content.decode("utf-8").strip().split("\n")
        features = []

        for line in lines:
            features.append([float(v) for v in line.strip().split()])

        return features

    def get_data(self, archive, name):
        """ To shorten duplicate code for name=train or name=test cases """
        def get_data_single(f):
            return self.get_feature(self.get_file_in_archive(archive,
                "UCI HAR Dataset/"+f))

        data = [
            get_data_single(name+"/Inertial Signals/body_acc_x_"+name+".txt"),
            get_data_single(name+"/Inertial Signals/body_acc_y_"+name+".txt"),
            get_data_single(name+"/Inertial Signals/body_acc_z_"+name+".txt"),
            get_data_single(name+"/Inertial Signals/body_gyro_x_"+name+".txt"),
            get_data_single(name+"/Inertial Signals/body_gyro_y_"+name+".txt"),
            get_data_single(name+"/Inertial Signals/body_gyro_z_"+name+".txt"),
            get_data_single(name+"/Inertial Signals/total_acc_x_"+name+".txt"),
            get_data_single(name+"/Inertial Signals/total_acc_y_"+name+".txt"),
            get_data_single(name+"/Inertial Signals/total_acc_z_"+name+".txt"),
        ]

        labels = get_data_single(name+"/y_"+name+".txt")

        subjects = get_data_single(name+"/subject_"+name+".txt")

        data = np.array(data, dtype=np.float32)
        labels = np.squeeze(np.array(labels, dtype=np.float32))
        # Squeeze so we can easily do selection on this later on
        subjects = np.squeeze(np.array(subjects, dtype=np.float32))

        # Transpose from [features, examples, time_steps] to
        # [examples, time_steps (128), features (9)]
        data = np.transpose(data, axes=[1, 2, 0])

        return data, labels, subjects

    def load_file(self, filename):
        """ Load ZIP file containing all the .txt files """
        with zipfile.ZipFile(filename, "r") as archive:
            train_data, train_labels, train_subjects = self.get_data(archive, "train")
            test_data, test_labels, test_subjects = self.get_data(archive, "test")

        all_data = np.vstack([train_data, test_data]).astype(np.float32)
        all_labels = np.hstack([train_labels, test_labels]).astype(np.float32)
        all_subjects = np.hstack([train_subjects, test_subjects]).astype(np.float32)

        # All data if no selection
        if self.users is None:
            return all_data, all_labels

        # Otherwise, select based on the desired users
        data = []
        labels = []

        for user in self.users:
            selection = all_subjects == user
            data.append(all_data[selection])
            current_labels = all_labels[selection]
            labels.append(current_labels)

        x = np.vstack(data).astype(np.float32)
        y = np.hstack(labels).astype(np.float32)

        # print("Selected data:", self.users)
        # print(x.shape, y.shape)

        return x, y

    def load(self):
        dataset_fp = self.download()
        x, y = self.load_file(dataset_fp)
        train_data, train_labels, test_data, test_labels = \
            self.train_test_split(x, y)

        # Only one modality
        return [train_data], train_labels, [test_data], test_labels

    def process(self, data, labels):
        # Index one
        labels = labels - 1
        return super().process(data, labels)


@register_dataset("ucihhar")
class UciHHarBase(Dataset):
    """
    Loads Heterogeneity Human Activity Recognition (HHAR) dataset
    http://archive.ics.uci.edu/ml/datasets/Heterogeneity+Activity+Recognition
    """
    feature_names = [[
        "acc_x", "acc_y", "acc_z",
        # "gyro_x", "gyro_y", "gyro_z",
    ]]
    num_classes = 6
    class_labels = [
        "bike", "sit", "stand", "walk", "stairsup", "stairsdown",
    ]  # we throw out "null"
    num_modalities = 1
    window_size = 128  # to be relatively similar to HAR
    window_overlap = False
    users = zero_to_n(8)  # 9 people

    def __init__(self, users, *args, **kwargs):
        self.users = users
        super().__init__(UciHHarBase.num_classes, UciHHarBase.class_labels,
            UciHHarBase.num_modalities,
            UciHHarBase.window_size, UciHHarBase.window_overlap,
            UciHHarBase.feature_names, *args, **kwargs)

    def download(self):
        (dataset_fp,) = self.download_dataset(["Activity%20recognition%20exp.zip"],
            "https://archive.ics.uci.edu/ml/machine-learning-databases/00344/")
        return dataset_fp

    def read_file(self, content):
        """ Read the CSV file """
        lines = content.decode("utf-8").strip().split("\n")
        data_x = []
        data_label = []
        data_subject = []
        users = ["a", "b", "c", "d", "e", "f", "g", "h", "i"]

        for line in lines:
            index, arrival, creation, x, y, z, user, \
                model, device, label = line.strip().split(",")

            # Skip the header (can't determine user if invalid)
            if index == "Index":
                continue

            user = users.index(user)  # letter --> number

            # Skip users we don't care about and data without a label
            if user in self.users and label != "null":
                #index = int(index)
                #arrival = float(arrival)
                #creation = float(creation)
                x = float(x)
                y = float(y)
                z = float(z)
                label = self.class_labels.index(label)  # name --> number

                data_x.append((x, y, z))
                data_label.append(label)
                data_subject.append(user)

        data_x = np.array(data_x, dtype=np.float32)
        data_label = np.array(data_label, dtype=np.float32)
        data_subject = np.array(data_subject, dtype=np.float32)

        return data_x, data_label, data_subject

    def get_data(self, archive, name):
        # In their paper, looks like they only did either accelerometer or
        # gyroscope, not aligning them by the creation timestamp. For them the
        # accelerometer data worked better, so we'll just use that for now.
        return self.read_file(self.get_file_in_archive(archive,
                "Activity recognition exp/"+name+"_accelerometer.csv"))

    def load_file(self, filename):
        """ Load ZIP file containing all the .txt files """
        with zipfile.ZipFile(filename, "r") as archive:
            # For now just use phone data since the positions may differ too much
            all_data, all_labels, all_subjects = self.get_data(archive, "Phones")

            # phone_data, phone_labels, phone_subjects = self.get_data(archive, "Phone")
            # watch_data, watch_labels, watch_subjects = self.get_data(archive, "Watch")

        # all_data = np.vstack([phone_data, watch_data]).astype(np.float32)
        # all_labels = np.hstack([phone_labels, watch_labels]).astype(np.float32)
        # all_subjects = np.hstack([phone_subjects, watch_subjects]).astype(np.float32)

        # Otherwise, select based on the desired users
        data = []
        labels = []

        for user in self.users:
            # Load this user's data
            selection = all_subjects == user
            current_data = all_data[selection]
            current_labels = all_labels[selection]
            assert len(current_labels) > 0, "Error: no data for user "+str(user)

            # Split into windows
            current_data, current_labels = self.create_windows(current_data,
                current_labels, self.window_size, self.window_overlap)

            # Save
            data.append(current_data)
            labels.append(current_labels)

        x = np.vstack(data).astype(np.float32)
        y = np.hstack(labels).astype(np.float32)

        # print("Selected data:", self.users)
        # print(x.shape, y.shape)

        return x, y

    def load(self):
        dataset_fp = self.download()
        x, y = self.load_file(dataset_fp)
        train_data, train_labels, test_data, test_labels = \
            self.train_test_split(x, y)

        # Only one modality
        return [train_data], train_labels, [test_data], test_labels


class WisdmBase(Dataset):
    """
    Base class for the WISDM datasets
    http://www.cis.fordham.edu/wisdm/dataset.php
    """
    feature_names = [[
        "acc_x", "acc_y", "acc_z",
    ]]
    window_size = 128  # similar to HAR
    window_overlap = False

    def __init__(self, users, num_classes, class_labels, num_modalities,
            *args, class_labels_map=None, **kwargs):
        self.users = users
        self.class_labels_map = class_labels_map
        super().__init__(num_classes, class_labels, num_modalities,
            WisdmBase.window_size, WisdmBase.window_overlap,
            WisdmBase.feature_names, *args, **kwargs)
        # Override and set these
        #self.filename_prefix = ""
        #self.download_filename = ""

    def download(self):
        (dataset_fp,) = self.download_dataset([self.download_filename],
            "http://www.cis.fordham.edu/wisdm/includes/datasets/latest/")
        return dataset_fp

    def read_data(self, lines, user_list):
        """ Read the raw data CSV file """
        data_x = []
        data_label = []
        data_subject = []

        for line in lines:
            parts = line.strip().replace(";", "").split(",")

            # For some reason there's blank rows in the data, e.g.
            # a bunch of lines like "577,,;"
            # Though, allow 7 since sometimes there's an extra comma at the end:
            # "21,Jogging,117687701514000,3.17,9,1.23,;"
            if len(parts) != 6 and len(parts) != 7:
                continue

            # Skip if x, y, or z is blank
            if parts[3] == "" or parts[4] == "" or parts[5] == "":
                continue

            user = int(parts[0])

            # Skip users that may not have enough data
            if user in user_list:
                user = user_list.index(user)  # non-consecutive to consecutive

                # Skip users we don't care about
                if user in self.users:
                    x = float(parts[3])
                    y = float(parts[4])
                    z = float(parts[5])
                    label = parts[1]
                    if self.class_labels_map is not None:
                        label = self.class_labels_map[label]
                    label = self.class_labels.index(label)  # name --> number

                    data_x.append((x, y, z))
                    data_label.append(label)
                    data_subject.append(user)

        data_x = np.array(data_x, dtype=np.float32)
        data_label = np.array(data_label, dtype=np.float32)
        data_subject = np.array(data_subject, dtype=np.float32)

        return data_x, data_label, data_subject

    def read_user_list(self, lines, min_test_samples=30):
        """ Read first column of the CSV file to get a unique list of uid's
        Also, skip users with too few samples """
        user_sample_count = {}

        for line in lines:
            parts = line.strip().split(",")

            # There's some lines without the right number of parts, e.g. blank
            if len(parts) != 6 and len(parts) != 7:
                continue

            # Skip if x, y, or z is blank
            if parts[3] == "" or parts[4] == "" or parts[5] == "":
                continue

            uid = int(parts[0])

            # There are duplicates in the file for some reason (so, either the
            # same person or it's not truly unique)
            if uid not in user_sample_count:
                user_sample_count[uid] = 0
            else:
                user_sample_count[uid] += 1

        # Remove users with too few samples
        user_list = []

        # How many samples we need: to stratify the sklearn function says
        # The test_size = A should be greater or equal to the number of classes = B
        # x/128*.2 > 6 classes
        # x > 6*128/.2
        # Though, actually, just set the minimum test samples. It's probably not
        # enough to have only 7...
        test_percentage = 0.20  # default
        #min_samples = int(len(self.class_labels)*self.window_size/test_percentage)
        min_samples = int(min_test_samples*self.window_size/test_percentage)

        for user, count in user_sample_count.items():
            if count > min_samples:
                user_list.append(user)

        # Data isn't sorted by user in the file
        user_list.sort()

        return user_list

    def get_lines(self, archive, name):
        """ Open and load file in tar file, get lines from file """
        f = archive.extractfile(self.filename_prefix+name)

        if f is None:
            return None

        return f.read().decode("utf-8").strip().split("\n")

    def load_file(self, filename):
        """ Load desired participants' data """
        # Get data
        with tarfile.open(filename, "r") as archive:
            raw_data = self.get_lines(archive, "raw.txt")

        # Some of the data doesn't have a uid in the demographics file? So,
        # instead just get the user list from the raw data. Also, one person
        # have very little data, so skip them (e.g. one person only has 25
        # samples, which is only 0.5 seconds of data -- not useful).
        user_list = self.read_user_list(raw_data)

        #print("Number of users:", len(user_list))

        # For now just use phone data since the positions may differ too much
        all_data, all_labels, all_subjects = self.read_data(raw_data, user_list)

        # Otherwise, select based on the desired users
        data = []
        labels = []

        for user in self.users:
            # Load this user's data
            selection = all_subjects == user
            current_data = all_data[selection]
            current_labels = all_labels[selection]
            assert len(current_labels) > 0, "Error: no data for user "+str(user)

            # Split into windows
            current_data, current_labels = self.create_windows(current_data,
                current_labels, self.window_size, self.window_overlap)

            # Save
            data.append(current_data)
            labels.append(current_labels)

        x = np.vstack(data).astype(np.float32)
        y = np.hstack(labels).astype(np.float32)

        # print("Selected data:", self.users)
        # print(x.shape, y.shape)

        return x, y

    def load(self):
        dataset_fp = self.download()
        x, y = self.load_file(dataset_fp)
        train_data, train_labels, test_data, test_labels = \
            self.train_test_split(x, y)

        # Only one modality
        return [train_data], train_labels, [test_data], test_labels


@register_dataset("wisdm_at")
class WisdmAtBase(WisdmBase):
    """
    Loads Actitracker dataset
    http://www.cis.fordham.edu/wisdm/dataset.php#actitracker (note: click
    on Actitracker link on left menu)
    """
    num_classes = 6
    class_labels = [
        "Walking", "Jogging", "Stairs", "Sitting", "Standing", "LyingDown",
    ]
    num_modalities = 1
    users = zero_to_n(50)  # 51 people

    def __init__(self, users, *args, **kwargs):
        self.filename_prefix = "home/share/data/public_sets/WISDM_at_v2.0/WISDM_at_v2.0_"
        self.download_filename = "WISDM_at_latest.tar.gz"
        super().__init__(users,
            WisdmAtBase.num_classes, WisdmAtBase.class_labels,
            WisdmAtBase.num_modalities, *args, **kwargs)


@register_dataset("wisdm_ar")
class WisdmArBase(WisdmBase):
    """
    Loads WISDM Activity prediction/recognition dataset
    http://www.cis.fordham.edu/wisdm/dataset.php
    """
    num_classes = 6
    class_labels = [
        "Walking", "Jogging", "Sitting", "Standing", "Upstairs", "Downstairs",
    ]
    num_modalities = 1
    users = zero_to_n(32)  # 33 people

    def __init__(self, users, *args, **kwargs):
        self.filename_prefix = "WISDM_ar_v1.1/WISDM_ar_v1.1_"
        self.download_filename = "WISDM_ar_latest.tar.gz"
        super().__init__(users,
            WisdmArBase.num_classes, WisdmArBase.class_labels,
            WisdmArBase.num_modalities, *args, **kwargs)


@register_dataset("myo")
class Myo(Dataset):
    """
    Loads Myo dataset
    https://arxiv.org/pdf/1801.07756.pdf
    https://github.com/UlysseCoteAllard/MyoArmbandDataset
    """
    feature_names = [[
        "ch1", "ch2", "ch3", "ch4", "ch5", "ch6", "ch7", "ch8",
    ]]
    num_classes = 7
    class_labels = [
        "Neutral",
        "RadialDeviation",
        "WristFlexion",
        "UlnarDeviation",
        "WristExtension",
        "HandClose",
        "HandOpen",
    ]
    num_modalities = 1
    window_size = 52  # 200 Hz, no effect, windows created in load_myo_data
    window_overlap = False  # No effect, overlap applied in load_myo_data
    users = zero_to_n(39)  # 40 people
    target_users = list(range(22, 39+1))  # but, only use 22-39 as targets

    def __init__(self, users, *args, **kwargs):
        self.users = users
        super().__init__(Myo.num_classes, Myo.class_labels,
            Myo.num_modalities, Myo.window_size, Myo.window_overlap,
            Myo.feature_names, *args, **kwargs)

        assert Myo.num_classes == load_myo_data.number_of_classes
        assert len(Myo.class_labels) == Myo.num_classes

    def get_data(self, folder, data_type):
        """ Load one user's data, based on load_myo_data.read_data() """
        user_examples = []
        user_labels = []

        for i in range(load_myo_data.number_of_classes * 4):
            data_file = os.path.join(folder, data_type, "classe_{}.dat".format(i))
            data_read_from_file = np.fromfile(data_file, dtype=np.int16)
            data_read_from_file = np.array(data_read_from_file, dtype=np.float32)
            dataset_example = load_myo_data.format_data_to_train(data_read_from_file)
            labels = (i % load_myo_data.number_of_classes) + np.zeros(dataset_example.shape[0])

            # print("{}:".format(i),
            #     "class", i%load_myo_data.number_of_classes,
            #     "iter", i//load_myo_data.number_of_classes,
            #     "-", dataset_example.shape, labels.shape)

            user_examples.append(dataset_example)
            user_labels.append(labels)

        user_examples, user_labels = load_myo_data.shift_electrodes(user_examples, user_labels)

        # Convert from list of examples to one matrix
        user_examples = np.vstack(user_examples).astype(np.float32)
        user_labels = np.hstack(user_labels).astype(np.float32)

        # Remove extra 1 dimension, e.g. [examples, 1, features, time_steps]
        user_examples = np.squeeze(user_examples, axis=1)

        # Transpose from [examples, features, time_steps] to
        # [examples, time_steps, features]
        user_examples = np.transpose(user_examples, axes=[0, 2, 1])

        return user_examples, user_labels

    def load(self):
        """
        Load desired participants' data

        We combine pre-training and evaluation datasets since we're doing
        unsupervised domain adaptation rather than supervised domain adaptation
        (i.e., pre-training and then fine-tuning with some target-domain labeled
        data like they did)

        Numbering:
            0 - PreTrainingDataset/Female0
            1 - PreTrainingDataset/Female1
            ...
            9 - PreTrainingDataset/Female9

            10 - PreTrainingDataset/Male0
            11 - PreTrainingDataset/Male1
            ...
            21 - PreTrainingDataset/Male11

            22 - EvaluationDataset/Female0
            23 - EvaluationDataset/Female1

            24 - EvaluationDataset/Male0
            25 - EvaluationDataset/Male1
            ...
            39 - EvaluationDataset/Male15

        All the pre-training datasets are included in training data, i.e. those
        test sets are empty. For evaluation datasets, the "training0" data is
        the train set and the "Test0" and "Test1" are combined as the test set.

        Each training set is later further split into train/valid when creating
        the tfrecord files.

        Note: only use 22-39 for the target domains!
        """
        # Locations of data, within git submodule
        data_folder = os.path.join("datasets", "MyoArmbandDataset")
        pre_folder = os.path.join(data_folder, "PreTrainingDataset")
        eval_folder = os.path.join(data_folder, "EvaluationDataset")

        # Get (folder, is_evaluation) pair from the user id
        folder_map = {}
        for i in range(0, 9+1):
            folder_map[i] = (os.path.join(pre_folder, "Female{}".format(i)), False)
        for i in range(10, 21+1):
            folder_map[i] = (os.path.join(pre_folder, "Male{}".format(i-10)), False)
        for i in range(22, 23+1):
            folder_map[i] = (os.path.join(eval_folder, "Female{}".format(i-22)), True)
        for i in range(24, 39+1):
            folder_map[i] = (os.path.join(eval_folder, "Male{}".format(i-24)), True)

        data_train = []
        labels_train = []
        data_test = []
        labels_test = []

        for user in self.users:
            folder, is_evaluation = folder_map[user]
            user_data_train, user_labels_train = self.get_data(folder, "training0")

            user_data_test = None
            user_labels_test = None

            if is_evaluation:
                user_data_test0, user_labels_test0 = self.get_data(folder, "Test0")
                user_data_test1, user_labels_test1 = self.get_data(folder, "Test1")
                user_data_test = np.vstack([user_data_test0, user_data_test1]).astype(np.float32)
                user_labels_test = np.hstack([user_labels_test0, user_labels_test1]).astype(np.float32)

            data_train.append(user_data_train)
            labels_train.append(user_labels_train)

            if user_data_test is not None:
                data_test.append(user_data_test)
                labels_test.append(user_labels_test)

        x_train = np.vstack(data_train).astype(np.float32)
        y_train = np.hstack(labels_train).astype(np.float32)

        if len(data_test) > 0 and len(labels_test) > 0:
            x_test = np.vstack(data_test).astype(np.float32)
            y_test = np.hstack(labels_test).astype(np.float32)
        else:
            x_test = np.array([], dtype=np.float32)
            y_test = np.array([], dtype=np.float32)

        # print("Selected data:", self.users)
        # print(x_train.shape, y_train.shape)
        # print(x_test.shape, y_test.shape)

        # Only one modality
        return [x_train], y_train, [x_test], y_test


class NinaProBase(Dataset):
    """ NinaPro datasets

    Base class handling splitting into windows, train/test, loading from the
    .mat files, etc.

    - include_rest: If include_rest=True then num_classes should be one more
      than if include_rest=False (i.e., num_classes needs to include the "rest"
      if we want it included).
    - channel_subset: (start, end) inclusive, e.g. if there's 16 channels 0-15
      and you want to select the last 8, then pass (8, 15)
    - balance_rest: If include_rest, then subsample this data to make it
      approximately balanced with the other classes (i.e. has the average number
      of windows that the others have).
    - sort_electrodes: Only use this if using the like-Myo dataset. It depends
      on include_rest=True and label_subset being the appropriate subset of
      DB5 to match Myo's class labels.
    """
    num_modalities = 1

    def __init__(self, users, num_classes, class_labels, feature_names,
            window_size, window_overlap, which_db, which_ex,
            include_rest=False, label_subset=None, channel_subset=None,
            balance_rest=True, shift_electrodes=False, *args, **kwargs):
        self.users = users
        self.which_db = which_db
        self.which_ex = which_ex
        self.include_rest = include_rest
        self.label_subset = label_subset
        self.channel_subset = channel_subset
        self.balance_rest = balance_rest
        self.shift_electrodes = shift_electrodes
        super().__init__(num_classes, class_labels,
            NinaProBase.num_modalities,
            window_size, window_overlap,
            feature_names, *args, **kwargs)

    def download(self):
        raise NotImplementedError("must override download()")

    def archive_path(self, subject):
        """ Get file path of the .mat file in the zip file for this user """
        db = self.which_db
        ex = self.which_ex

        # The filenames are indexed starting at 1 not zero
        subject += 1

        if db == 1:
            template = "DB1_s{subject}/S{subject}_A1_E{exercise}.mat"
        elif db == 5:
            template = "s{subject}/S{subject}_E{exercise}_A1.mat"
        else:
            raise NotImplementedError("currently only support NinaPro DB1 and DB5")

        return template.format(subject=subject, exercise=ex)

    def db_repetitions(self):
        db = self.which_db

        # Try to get around 80% / 20% to match what we do for the other datasets
        if db == 1:
            train = [1, 2, 3, 4, 5, 6, 7, 8]
            test = [9, 10]
        elif db == 5:
            train = [1, 2, 3, 4, 5]
            test = [6]
        else:
            raise NotImplementedError("currently only support NinaPro DB1 and DB5")

        return train, test

    def get_data(self, archive, filename, repetitions, random_seed=42):
        """ Open .mat file in zip file, then load contents, get desired data,
        split into multiple sliding windows """
        # Get data from .mat inside the .zip file
        with archive.open(filename) as fp:
            mat = scipy.io.loadmat(fp)

            xs = mat["emg"]

            # Channel subset
            if self.channel_subset is not None:
                assert isinstance(self.channel_subset, tuple) \
                    and len(self.channel_subset) == 2, \
                    "channel_subset should be of the form (start_ch, end_ch)"

                channel_start, channel_end = self.channel_subset
                # Select start to end, inclusive, thus end+1
                xs = xs[:, channel_start:channel_end+1]

            # They say restimulus is the "the corrected stimulus, processed with
            # movement detection algorithms" whereas stimulus is "the original
            # label of the movement repeated by the subject." Thus, it sounds
            # like we should use restimulus since it's the "correct" one.
            ys = mat["restimulus"]

            # Repetition, but use rerepetition similar to why we use restimulus
            reps = mat["rerepetition"]

            # Set of labels - to verify our label_subset consists of valid
            # labels (if specified)
            label_set = np.unique(mat["restimulus"])

        # Load the x/y data for the desired set of repetitions
        data = []
        labels = []
        windows_per_label = []

        # We handle rest differently because otherwise there's way more "rest"
        # than the other classes
        data_rest = []
        labels_rest = []

        if self.label_subset is not None:
            assert isinstance(self.label_subset, list), \
                "label_subset should be a list"
            assert all([
                label in label_set for label in self.label_subset
            ]), "not all labels in label_subset are found in the file"

            all_classes = self.label_subset
        else:
            all_classes = list(range(self.num_classes))

        if self.include_rest:
            label_offset = 0
        else:
            # If we're excluding rest, then make sure label_subset does not
            # contain 0
            assert self.label_subset is None or 0 not in self.label_subset, \
                "label_subset should not contain label 0 (rest) if include_rest=False"

            label_offset = 1

        # We use label to get from the file, but output with label_index
        # so our final labels are 0, 1, ..., num_classes-1 even if we take a
        # subset of the labels.
        for label_index, label in enumerate(all_classes):
            windows_per_this_label = 0

            for rep in repetitions:
                # Get just the data with this label for this repetition
                #
                # Note: we skip rest, so label 0 is the first movement not rest,
                # if not include_rest.
                wh = np.where(np.squeeze(np.logical_and(
                    ys == label+label_offset, reps == rep), axis=1))
                x = xs[wh]

                # Within each repetition for each class, split into
                # overlapping windows. We don't have to worry about
                # train/test overlap since they use non-overlapping
                # repetitions.
                x = self.create_windows_x(x, self.window_size, self.window_overlap)
                y = [label_index]*len(x)

                windows_per_this_label += len(x)

                if self.include_rest and self.balance_rest and label == 0:
                    data_rest.append(x)
                    labels_rest.append(y)
                else:
                    data.append(x)
                    labels.append(y)

            # Exclude "rest" from this average, since we're using this to
            # balance the "rest" class data
            if self.include_rest and self.balance_rest and label != 0:
                windows_per_label.append(windows_per_this_label)

        # Take a subset of "rest" if balance_rest
        if self.include_rest and self.balance_rest:
            avg_windows_others = np.mean(windows_per_label).astype(np.int32)
            # print("Taking only first", avg_windows_others, "of rest")

            # Concat all labels/reps, otherwise if we take the subset first we
            # essentially get all the data
            data_rest = np.vstack(data_rest).astype(np.float32)
            labels_rest = np.hstack(labels_rest).astype(np.float32)

            # Shuffle both together; we don't want to always get just the first
            # "rest" instances. Also, make this repeatable.
            p = np.random.RandomState(seed=random_seed).permutation(len(data_rest))
            data_rest = data_rest[p]
            labels_rest = labels_rest[p]

            # Limit the number, also put at the beginning in case
            # shift_electrodes
            data.append(data_rest[:avg_windows_others])
            labels.append(labels_rest[:avg_windows_others])

        # Shift electrodes if desired -- *must* be the like-Myo dataset since
        # this assumes Myo dataset labels and channels
        if self.shift_electrodes:
            # Put into the format that load_myo_data.shift_electrodes assumes
            data_for_shift = []
            labels_for_shift = []
            assert len(data) == len(labels)

            for i in range(len(data)):
                # Transpose from [examples, time_steps, features] to
                # [examples, features, time_steps]
                x = np.transpose(data[i], axes=[0, 2, 1])

                # Add extra dimension: [examples, 1, features, time_steps]
                x = np.expand_dims(x, axis=1)

                data_for_shift.append(x)
                labels_for_shift.append(labels[i])

                # print("{}:".format(i), "-", x.shape, np.array(labels[i]).shape)

            # Perform electrode shift
            shifted_data, shifted_labels = load_myo_data.shift_electrodes(data_for_shift, labels_for_shift)

            # Put back in our format
            assert len(shifted_data) == len(shifted_labels)
            assert len(shifted_data) == len(data)
            data = []
            labels = []

            for i in range(len(shifted_data)):
                # Remove extra 1 dimension, e.g. [examples, 1, features, time_steps]
                x = np.squeeze(shifted_data[i], axis=1)

                # Transpose from [examples, features, time_steps] to
                # [examples, time_steps, features]
                x = np.transpose(x, axes=[0, 2, 1])

                data.append(x)
                labels.append(shifted_labels[i])

        data = np.vstack(data).astype(np.float32)
        labels = np.hstack(labels).astype(np.float32)

        # Check balancing
        # print("Balance")
        # for i in range(self.num_classes):
        #     print("  Label {}:".format(i), sum(labels == i))
        # print()

        return data, labels

    def get_data_set(self, files, repetitions):
        data = []
        labels = []

        for user, f in files:
            with zipfile.ZipFile(f, "r") as archive:
                current_data, current_labels = self.get_data(archive,
                    self.archive_path(user), repetitions)
                data.append(current_data)
                labels.append(current_labels)

        data = np.vstack(data).astype(np.float32)
        labels = np.hstack(labels).astype(np.float32)

        return data, labels

    def load(self):
        dataset_fp = self.download()
        files = [(i, dataset_fp[i]) for i in self.users]

        # Get data from the .mat files for the desired user(s)
        train_repetitions, test_repetitions = self.db_repetitions()
        train_data, train_labels = self.get_data_set(files, train_repetitions)
        test_data, test_labels = self.get_data_set(files, test_repetitions)

        # print("Train:", train_data.shape, train_labels.shape)
        # print("Test:", test_data.shape, test_labels.shape)

        # Only one modality
        return [train_data], train_labels, [test_data], test_labels


# class NinaProExercise1Base(NinaProBase):
#     """ NinaPro datasets, exercise 1

#     Base class on top of NinaProBase which handles the class labels
#     """
#     num_classes = 12  # excluding rest
#     class_labels = [
#         "class{}".format(i) for i in range(12)
#     ]

#     def __init__(self, users, feature_names, window_size, window_overlap,
#             which_db, *args, **kwargs):
#         super().__init__(users,
#             NinaProExercise1Base.num_classes,
#             NinaProExercise1Base.class_labels,
#             feature_names, window_size, window_overlap, which_db,
#             which_ex=1, *args, **kwargs)


class NinaProExercise2Base(NinaProBase):
    """ NinaPro datasets, exercise 2

    Base class on top of NinaProBase which handles the class labels
    """
    num_classes = 17  # excluding rest
    class_labels = [
        "class{}".format(i) for i in range(17)
    ]

    def __init__(self, users, feature_names, window_size, window_overlap,
            which_db, *args, **kwargs):
        super().__init__(users,
            NinaProExercise2Base.num_classes,
            NinaProExercise2Base.class_labels,
            feature_names, window_size, window_overlap, which_db,
            which_ex=2, *args, **kwargs)


class NinaProExercise2LikeMyo(NinaProBase):
    """ NinaPro dataset exercise 2 like Myo
    As in the Myo paper: https://arxiv.org/pdf/1801.07756.pdf

    - Only exercise 2
    - Only classes that agree with the Myo dataset
    - Include rest
    - Perform electrode shifting like in Myo dataset (unless shift_electrods=False)

    Base class on top of NinaProBase which handles the class labels, etc.
    """
    num_classes = 7
    class_labels = [
        "Neutral",
        "RadialDeviation",
        "WristFlexion",
        "UlnarDeviation",
        "WristExtension",
        "HandClose",
        "HandOpen",
    ]

    def __init__(self, users, feature_names, window_size, window_overlap,
            which_db, shift_electrodes=True, *args, **kwargs):
        # Note: for each exercise, the labels are given in Figure 2 on
        # https://www.nature.com/articles/sdata201453
        super().__init__(users,
            NinaProExercise2LikeMyo.num_classes,
            NinaProExercise2LikeMyo.class_labels,
            feature_names, window_size, window_overlap, which_db,
            which_ex=2, include_rest=True, label_subset=[
                0,   # Neutral
                15,  # "Wrist radial deviation"
                13,  # "Wrist flexion"
                16,  # "Wrist ulnar deviation"
                14,  # "Wrist extension"
                6,   # "Fingers flexed together in fist" (hand closed?)
                5,   # "Abduction of all fingers" (hand open?)
            ], shift_electrodes=shift_electrodes, *args, **kwargs)

# class NinaProExercise3Base(NinaProBase):
#     """ NinaPro datasets, exercise 3

#     Base class on top of NinaProBase which handles the class labels
#     """
#     num_classes = 23  # excluding rest
#     class_labels = [
#         "class{}".format(i) for i in range(23)
#     ]

#     def __init__(self, users, feature_names, window_size, window_overlap,
#             which_db, *args, **kwargs):
#         super().__init__(users,
#             NinaProExercise3Base.num_classes,
#             NinaProExercise3Base.class_labels,
#             feature_names, window_size, window_overlap, which_db,
#             which_ex=3, *args, **kwargs)


# class NinaProDB1Base:
#     """
#     NinaPro DB1
#     https://datadryad.org/stash/dataset/doi:10.5061/dryad.1k84r
#     """
#     feature_names = [[
#         "ch1", "ch2", "ch3", "ch4", "ch5",
#         "ch6", "ch7", "ch8", "ch9", "ch10",
#     ]]

#     # Window length/overlap like in https://arxiv.org/pdf/1801.07756.pdf
#     window_size = 26  # 100Hz * 0.260s (260ms) = 26 samples
#     window_overlap = 2  # 26 - 100*.235 = 2.5, so round down to 2

#     users = zero_to_n(26)  # 27 people

#     def __init__(self, users, *args, **kwargs):
#         super().__init__(users, NinaProDB1Base.feature_names,
#             NinaProDB1Base.window_size, NinaProDB1Base.window_overlap,
#             which_db=1, *args, **kwargs)

#     def download(self):
#         """
#         How to get filenames

#         Copy Inner HTML of the element with "December 19, 2014" expanded below
#         the "Download dataset" button on:
#         https://datadryad.org/stash/dataset/doi:10.5061/dryad.1k84r

#         cat the_html.txt | grep downloads | \
#             sed -re 's/.*DB1_s([0-9]+).*file_stream\/([0-9]+).*/\1, \2/g' | \
#             sort -n | sed -re 's/([0-9]+), ([0-9]+)/("NinaPro_DB1_s\1.zip", "\2"),/g'
#         """
#         dataset_fps = self.download_dataset([
#             ("NinaPro_DB1_s1.zip", "42532"),
#             ("NinaPro_DB1_s2.zip", "42534"),
#             ("NinaPro_DB1_s3.zip", "42536"),
#             ("NinaPro_DB1_s4.zip", "42538"),
#             ("NinaPro_DB1_s5.zip", "42540"),
#             ("NinaPro_DB1_s6.zip", "42542"),
#             ("NinaPro_DB1_s7.zip", "42544"),
#             ("NinaPro_DB1_s8.zip", "42546"),
#             ("NinaPro_DB1_s9.zip", "42548"),
#             ("NinaPro_DB1_s10.zip", "42550"),
#             ("NinaPro_DB1_s11.zip", "42552"),
#             ("NinaPro_DB1_s12.zip", "42554"),
#             ("NinaPro_DB1_s13.zip", "42556"),
#             ("NinaPro_DB1_s14.zip", "42558"),
#             ("NinaPro_DB1_s15.zip", "42560"),
#             ("NinaPro_DB1_s16.zip", "42562"),
#             ("NinaPro_DB1_s17.zip", "42564"),
#             ("NinaPro_DB1_s18.zip", "42566"),
#             ("NinaPro_DB1_s19.zip", "42568"),
#             ("NinaPro_DB1_s20.zip", "42570"),
#             ("NinaPro_DB1_s21.zip", "42572"),
#             ("NinaPro_DB1_s22.zip", "42574"),
#             ("NinaPro_DB1_s23.zip", "42576"),
#             ("NinaPro_DB1_s24.zip", "42578"),
#             ("NinaPro_DB1_s25.zip", "42580"),
#             ("NinaPro_DB1_s26.zip", "42582"),
#             ("NinaPro_DB1_s27.zip", "42584"),
#         ], "https://datadryad.org/stash/downloads/file_stream/")
#         return dataset_fps


class NinaProDB5Base:
    """
    NinaPro DB5 - data from both lower/upper armbands
    https://zenodo.org/record/1000116
    """
    feature_names = [[
        "ch1", "ch2", "ch3", "ch4", "ch5",
        "ch6", "ch7", "ch8", "ch9", "ch10",
        "ch11", "ch12", "ch13", "ch14", "ch15",
        "ch16",
    ]]

    # Window length/overlap like in https://arxiv.org/pdf/1801.07756.pdf
    window_size = 52  # 200Hz * 0.260s (260ms) = 52 samples
    window_overlap = 5  # 52 - 200*.235 = 5

    users = zero_to_n(9)  # 10 people

    def __init__(self, users, *args, **kwargs):
        super().__init__(users, NinaProDB5Base.feature_names,
            NinaProDB5Base.window_size, NinaProDB5Base.window_overlap,
            which_db=5, *args, **kwargs)

    def download(self):
        dataset_fps = self.download_dataset([
            ("NinaPro_DB5_s{}.zip".format(i), "s{}.zip?download=1".format(i))
            for i in range(1, 10+1)
        ], "https://zenodo.org/record/1000116/files/")
        return dataset_fps


class NinaProDB5BaseLower:
    """
    NinaPro DB5 (only lower armband, for comparison with Myo)
    https://zenodo.org/record/1000116
    """
    feature_names = [[
        "ch1", "ch2", "ch3", "ch4", "ch5",
        "ch6", "ch7", "ch8",
    ]]

    # Window length/overlap like in https://arxiv.org/pdf/1801.07756.pdf
    window_size = 52  # 200Hz * 0.260s (260ms) = 52 samples
    window_overlap = 5  # 52 - 200*.235 = 5

    users = zero_to_n(9)  # 10 people

    def __init__(self, users, *args, **kwargs):
        super().__init__(users, NinaProDB5BaseLower.feature_names,
            NinaProDB5BaseLower.window_size, NinaProDB5BaseLower.window_overlap,
            which_db=5, channel_subset=(8, 15),
            *args, **kwargs)

    def download(self):
        dataset_fps = self.download_dataset([
            ("NinaPro_DB5_s{}.zip".format(i), "s{}.zip?download=1".format(i))
            for i in range(1, 10+1)
        ], "https://zenodo.org/record/1000116/files/")
        return dataset_fps


# @register_dataset("ninapro_db1_ex1")
# class NinaProDB1Ex1(NinaProDB1Base, NinaProExercise1Base):
#     pass


# @register_dataset("ninapro_db1_ex2")
# class NinaProDB1Ex2(NinaProDB1Base, NinaProExercise2Base):
#     pass


# @register_dataset("ninapro_db1_ex3")
# class NinaProDB1Ex3(NinaProDB1Base, NinaProExercise3Base):
#     pass


# @register_dataset("ninapro_db5_ex1")
# class NinaProDB5Ex1(NinaProDB5Base, NinaProExercise1Base):
#     pass


@register_dataset("ninapro_db5_ex2")
class NinaProDB5Ex2(NinaProDB5Base, NinaProExercise2Base):
    pass


# @register_dataset("ninapro_db5_ex3")
# class NinaProDB5Ex3(NinaProDB5Base, NinaProExercise3Base):
#     pass

@register_dataset("ninapro_db5_like_myo")
class NinaProDB5LikeMyo(NinaProDB5BaseLower, NinaProExercise2LikeMyo):
    pass


@register_dataset("ninapro_db5_like_myo_noshift")
class NinaProDB5LikeMyoNoShift(NinaProDB5BaseLower, NinaProExercise2LikeMyo):
    def __init__(self, users, *args, **kwargs):
        super().__init__(users, shift_electrodes=False, *args, **kwargs)


#
# Multivariate normal synthetic data
#
class MultivariateNormalSyntheticBase(Dataset):
    """ Multivariate normal synthetic datasets """
    num_modalities = 1
    # not time series
    window_size = -1
    window_overlap = None

    def __init__(self, users, num_classes, class_labels, features,
            *args, **kwargs):
        self.users = users
        super().__init__(
            num_classes,
            class_labels,
            MultivariateNormalSyntheticBase.num_modalities,
            MultivariateNormalSyntheticBase.window_size,
            MultivariateNormalSyntheticBase.window_overlap,
            features,
            *args, **kwargs)

    def load_file(self, filename, time_series=False):
        """
        Load CSV files in UCR time-series data format but with semicolons
        delimiting the features
        Returns:
            data - numpy array with data of shape (num_examples, time_steps, num_features)
            labels - numpy array with labels of shape: (num_examples, 1)

        Note: based on SyntheticDataBase()
        """
        with open(filename, "r") as f:
            data = []
            labels = []

            for line in f:
                parts = line.split(",")
                assert len(parts) >= 2, "must be at least a label and a data value"
                label = int(parts[0])
                values_str = parts[1:]
                values = []

                for value in values_str:
                    # If a time series, further split
                    if time_series:
                        features_str = value.split(";")
                        features = [float(v) for v in features_str]
                        values.append(features)
                    # Otherwise, there's just a few data values (i.e., x1 and x2)
                    else:
                        # If you want to visualize with
                        # python -m datasets.view_datasets --source=mvn_n4_l3_inter1_intra1_0
                        # values.append([float(value)])
                        # For actual experiments
                        values.append(float(value))

                labels.append(label)
                data.append(values)

        data = np.array(data, dtype=np.float32)
        labels = np.array(labels, dtype=np.float32)
        assert len(labels.shape) == 1, "incorrect label shape: not (n,)"
        assert all(labels >= 0), "labels should all be >= 0"

        return data, labels

    def load(self):
        # Get filename
        assert len(self.users) == 1, "currently only support one user"
        user = self.users[0]

        # Which synthetic data to use
        filename_base = "normal_n{}_l{}_inter{}_intra{}{}".format(
            self.num_sources, self.num_classes, self.inter, self.intra,
            "_" + self.suffix if self.suffix is not None else "")

        if self.raw:
            time_series = False
        else:
            filename_base += "_sine"
            time_series = True

        # We set target to be user 0, since there's only one target
        if user == 0:
            filename = filename_base + "_t"
        else:
            filename = filename_base + "_s{}".format(user-1)

        data_path = "datasets/synthetic"  # see synthetic_datasets.py
        train_filename = os.path.join(data_path, filename + "_TRAIN")
        test_filename = os.path.join(data_path, filename + "_TEST")

        # Load CSV-like files
        train_data, train_labels = self.load_file(train_filename, time_series)
        test_data, test_labels = self.load_file(test_filename, time_series)

        # Only one modality
        return [train_data], train_labels, [test_data], test_labels


def make_synthetic_datasets(num_sources, num_labels, inter, intra, suffix, raw=False):
    """ Reduces duplicate code, just with slight changes """

    class MultivariateNormalSyntheticDataset(MultivariateNormalSyntheticBase):
        # Changes
        num_classes = num_labels
        users = zero_to_n(num_sources)  # 0 is target, 1..n is sources

        # Same for all variations
        feature_names = [["x"]]
        class_labels = ["class{}".format(i) for i in range(num_classes)]
        target_users = [0]  # only allow this one target

        def __init__(self, users, *args, **kwargs):
            self.num_sources = num_sources  # same as above, exclude target
            self.inter = inter
            self.intra = intra
            self.suffix = suffix
            self.raw = raw
            super().__init__(users,
                MultivariateNormalSyntheticDataset.num_classes,
                MultivariateNormalSyntheticDataset.class_labels,
                MultivariateNormalSyntheticDataset.feature_names,
                **kwargs)

    dataset_name_suffix = "" if raw else "_sine"
    dataset = register_dataset("normal_n{}_l{}_inter{}_intra{}_{}{}".format(
        num_sources, num_labels, inter, intra, suffix, dataset_name_suffix
    ))(MultivariateNormalSyntheticDataset)

    return dataset


# Test inter/intra translate/rotate each separately; n=10, L=3
for suffix in ["5,0,0,0", "0,0.5,0,0"]:
    make_synthetic_datasets(4, 3, 0, 1, suffix)
    make_synthetic_datasets(4, 3, 1, 1, suffix)
    make_synthetic_datasets(4, 3, 2, 1, suffix)
    make_synthetic_datasets(12, 3, 0, 1, suffix)
    make_synthetic_datasets(12, 3, 1, 1, suffix)
    make_synthetic_datasets(12, 3, 2, 1, suffix)
for suffix in ["0,0,5,0", "0,0,0,0.5"]:
    make_synthetic_datasets(4, 3, 1, 0, suffix)
    make_synthetic_datasets(4, 3, 1, 1, suffix)
    make_synthetic_datasets(4, 3, 1, 2, suffix)
    make_synthetic_datasets(12, 3, 1, 0, suffix)
    make_synthetic_datasets(12, 3, 1, 1, suffix)
    make_synthetic_datasets(12, 3, 1, 2, suffix)


# Get datasets
def load(dataset_name_to_load, *args, **kwargs):
    """ Load a dataset based on the name (must be one of datasets.names()) """
    dataset_class = None
    dataset_object = None

    # Go through list of valid datasets, create the one this matches
    for name in list_datasets():
        for user in get_dataset_users(name):
            dataset_name = name+"_"+str(user)

            if dataset_name_to_load == dataset_name:
                dataset_class = get_dataset(name)
                dataset_object = call_dataset(name, users=[user],
                    *args, **kwargs)
                break

    if dataset_object is None:
        raise NotImplementedError("unknown dataset "+dataset_name_to_load)

    return dataset_object, dataset_class


# Get attributes: num_classes, class_labels (required in load_datasets.py)
def attributes(dataset_name_to_load):
    """ Get num_classes, class_labels for dataset (must be one of datasets.names()) """
    num_classes = None
    class_labels = None

    # Go through list of valid datasets, load attributes of the one this matches
    for name in list_datasets():
        for user in get_dataset_users(name):
            dataset_name = name+"_"+str(user)

            if dataset_name_to_load == dataset_name:
                d = get_dataset(name)
                num_classes = d.num_classes
                class_labels = d.class_labels
                break

    return num_classes, class_labels


# List of all valid dataset names
def names(single_modality=False):
    """ Returns list of all the available datasets to load with
    datasets.load(name) """
    datasets = []

    if single_modality:
        dataset_list = list_datasets_single_modality()
    else:
        dataset_list = list_datasets()

    for name in dataset_list:
        for user in get_dataset_users(name):
            datasets.append(name+"_"+str(user))

    return datasets


def main(argv):
    sd = load("ucihar_1")

    print("Source dataset")
    print(sd.train_data, sd.train_labels)
    print(sd.train_data.shape, sd.train_labels.shape)
    print(sd.test_data, sd.test_labels)
    print(sd.test_data.shape, sd.test_labels.shape)


if __name__ == "__main__":
    app.run(main)
