"""
Methods
"""
import random
import numpy as np
import tensorflow as tf
import tensorflow_addons as tfa

from absl import flags
from scipy.spatial.distance import euclidean
# from tqdm import tqdm

import models
import load_datasets

# from pool import run_job_pool
from pickle_data import load_pickle
from class_balance import class_balance

FLAGS = flags.FLAGS

flags.DEFINE_float("lr", 0.0001, "Learning rate for training")
flags.DEFINE_float("lr_domain_mult", 1.0, "Learning rate multiplier for training domain classifier")
flags.DEFINE_boolean("ensemble_same_data", False, "Train each model on the same batch of data, or if false use a different random batch for each model")
flags.DEFINE_float("similarity_weight", 10.0, "Weight for contrastive loss")
flags.DEFINE_integer("max_anchors", 0, "For cross-source self-supervision (v2), how many anchors to use; 0 = all of them")
flags.DEFINE_integer("max_positives", 10, "For cross-source self-supervision (v2), how many positives can be used for each anchor; 0 = all of them")
flags.DEFINE_integer("max_negatives", 40, "For cross-source self-supervision (v2), how many negatives can be used for each positive (actually relative to anchor, so probably make this bigger than max_positives); 0 = all of them")
flags.DEFINE_integer("contrastive_units", 128, "For cross-source self-supervision (v2), how many contrastive units to use")
flags.DEFINE_float("temperature", 0.1, "For cross-source self-supervision (v2), temperature parameter (tau)")
flags.DEFINE_float("ws_noise", 0, "When using weak supervision, how much to noise the target label proportions (simulates not completely accurate self-reporting)")

methods = {}


def register_method(name):
    """ Add method to the list of methods, e.g. add @register_method("name")
    before a class definition """
    assert name not in methods, "duplicate method named " + name

    def decorator(cls):
        methods[name] = cls
        return cls

    return decorator


def get_method(name, *args, **kwargs):
    """ Based on the given name, call the correct method """
    assert name in methods.keys(), \
        "Unknown method name " + name
    return methods[name](*args, **kwargs)


def list_methods():
    """ Returns list of all the available methods """
    return list(methods.keys())


class MethodBase:
    def __init__(self, source_datasets, target_dataset, model_name,
            *args, ensemble_size=1, trainable=True, moving_average=False,
            shared_modalities=None, share_most_weights=False,
            dataset_name=None, **kwargs):
        self.source_datasets = source_datasets
        self.target_dataset = target_dataset
        self.moving_average = moving_average
        self.ensemble_size = ensemble_size
        assert ensemble_size > 0, "ensemble_size should be >= 1"
        self.share_most_weights = share_most_weights  # for HeterogeneousBase
        self.dataset_name = dataset_name

        # Support multiple targets when we add that functionality
        self.num_source_domains = len(source_datasets)
        self.num_domains = len(source_datasets)

        if target_dataset is not None:
            if isinstance(target_dataset, list):
                self.num_domains += len(target_dataset)
            elif isinstance(target_dataset, load_datasets.Dataset):
                self.num_domains += 1
            else:
                raise NotImplementedError("target_dataset should be either one "
                    "load_datasets.Dataset() or a list of them, "
                    "but is "+str(target_dataset))

        if shared_modalities is not None:
            self.shared_modalities = [int(x) for x in shared_modalities.split(",")]
        else:
            # Use first modality, assuming there's only one
            # Note: this of course also requires that there is a target domain.
            source_num_modalities, target_num_modalities = self.get_num_modalities()
            assert source_num_modalities == 1, \
                "if multiple modalities, set shared_modalities"
            assert target_num_modalities == 1, \
                "if multiple modalities, set shared_modalities"

            self.shared_modalities = [0]

        # How to calculate the number of domain outputs
        self.domain_outputs = self.calculate_domain_outputs()

        # We need to know the num_classes for creating the model
        # We'll just pick the first source since we have to have at least one
        # source and we've already verified they're all the same in load_da()
        self.num_classes = source_datasets[0].num_classes

        # Needed for CoDATS label
        self.train_batch = source_datasets[0].train_batch

        # What we want in the checkpoint
        self.checkpoint_variables = {}

        # Initialize components -- support ensemble, training all simultaneously
        # I think will be faster / more efficient overall time-wise
        self.create_iterators()
        self.opt = [self.create_optimizers() for _ in range(ensemble_size)]
        self.model = [self.create_model(model_name) for _ in range(ensemble_size)]
        self.create_losses()

        # Checkpoint/save the model and optimizers
        for i, model in enumerate(self.model):
            self.checkpoint_variables["model_" + str(i)] = model

        for i, opt_dict in enumerate(self.opt):
            for name, opt in opt_dict.items():
                self.checkpoint_variables["opt_" + name + "_" + str(i)] = opt

        # Names of the losses returned in compute_losses
        self.loss_names = ["total"]

        # Should this method be trained (if not, then in main.py the config
        # is written and then it exits)
        self.trainable = trainable

    def get_num_modalities(self):
        """ Get the number of source/target modalities """
        source_num_modalities = self.source_datasets[0].num_modalities
        for other_source in range(1, len(self.source_datasets)):
            assert self.source_datasets[other_source].num_modalities \
                == source_num_modalities, \
                "sources with different num_modalities not supported yet"

        if self.target_dataset is not None:
            target_num_modalities = self.target_dataset.num_modalities
        else:
            target_num_modalities = None

        return source_num_modalities, target_num_modalities

    def calc_num_components(self):
        """
        We need a feature extractor for each modality. Take the max of the
        source or target number of modalities. Note: if for example the source
        has 2 and the target has 1, the lowest index source FE will be used
        for the target modality ("left to right" of sorts). Change the
        ordering with {source,target}_modality_subset if desired.

        We can't use len(self.shared_modalities) here since we need a FE even
        for the non-shared modalities.
        """
        source_num_modalities, target_num_modalities = self.get_num_modalities()

        if target_num_modalities is not None:
            num_feature_extractors = max(source_num_modalities,
                target_num_modalities)
        else:
            num_feature_extractors = source_num_modalities

        # However, for the domain classifiers, we need one per shared modality.
        num_domain_classifiers = len(self.shared_modalities)

        print("Creating", num_feature_extractors, "feature extractors")
        print("Creating", num_domain_classifiers, "domain classifiers")

        return num_feature_extractors, num_domain_classifiers

    def calculate_domain_outputs(self):
        """ Calculate the number of outputs for the domain classifier. By
        default it's the number of domains. However, for example, in domain
        generalization we ignore the target, so it'll actually be the number of
        source domains only, in which case override this function. """
        return self.num_domains

    def create_iterators(self):
        """ Get the source/target train/eval datasets """
        self.source_domain_ids = [x.domain_id for x in self.source_datasets]
        self.source_train_iterators = [iter(x.train) for x in self.source_datasets]
        self.source_train_eval_datasets = [x.train_evaluation for x in self.source_datasets]
        self.source_test_eval_datasets = [x.test_evaluation for x in self.source_datasets]

        if self.target_dataset is not None:
            self.target_domain_id = self.target_dataset.domain_id
            self.target_train_iterator = iter(self.target_dataset.train)
            self.target_train_eval_dataset = self.target_dataset.train_evaluation
            self.target_test_eval_dataset = self.target_dataset.test_evaluation
        else:
            self.target_domain_id = None
            self.target_train_iterator = None
            self.target_train_eval_dataset = None
            self.target_test_eval_dataset = None

    def create_optimizer(self, *args, **kwargs):
        """ Create a single optimizer """
        opt = tf.keras.optimizers.Adam(*args, **kwargs)

        if self.moving_average:
            opt = tfa.optimizers.MovingAverage(opt)

        return opt

    def create_optimizers(self):
        return {"opt": self.create_optimizer(learning_rate=FLAGS.lr)}

    def create_model(self, model_name):
        return models.BasicModel(self.num_classes, self.domain_outputs,
            model_name=model_name)

    def create_losses(self):
        self.task_loss = make_loss()

    @tf.function
    def get_next_train_data(self):
        """ Get next batch of training data """
        # Note we will use this same exact data in Metrics() as we use in
        # train_step()
        data_sources = [next(x) for x in self.source_train_iterators]
        data_target = next(self.target_train_iterator) \
            if self.target_train_iterator is not None else None
        return self.get_next_batch_both(data_sources, data_target)

    def domain_label(self, index, is_target):
        """ Default domain labeling. Indexes should be in [0,+inf) and integers.
        0 = target
        1 = source #0
        2 = source #1
        3 = source #2
        ...
        """
        if is_target:
            return 0
        else:
            return index+1

    @tf.function
    def get_next_batch_both(self, data_sources, data_target):
        """ Compile for training. Don't for evaluation (called directly,
        not this _both function). """
        data_sources = self.get_next_batch_multiple(data_sources, is_target=False)
        data_target = self.get_next_batch_single(data_target, is_target=True)
        return data_sources, data_target

    def get_next_batch_multiple(self, data, is_target):
        """
        Get next set of training data. data should be a list of data (probably
        something like [next(x) for x in iterators]).

        Returns: (
            [x_a1, x_a2, x_a3, ...],
            [y_a1, y_a2, y_a3, ...],
            [domain_a1, domain_a2, domain_a3, ...]
        )
        """
        if data is None:
            return None

        assert not is_target or len(data) == 1, \
            "only support one target at present"

        xs = []
        ys = []
        ds = []
        example_ids = []

        for i, (x, y, example_id) in enumerate(data):
            xs.append(x)
            ys.append(y)
            ds.append(tf.ones_like(y)*self.domain_label(index=i,
                is_target=is_target))
            example_ids.append(example_id)

        return (xs, ys, ds, example_ids)

    def get_next_batch_single(self, data, is_target, index=0):
        """
        Get next set of training data. data should be a single batch (probably
        something like next(iterator)). When processing target data, index
        must be 0 since we only support one target at the moment. However,
        during evaluation we evaluate each source's data individually so if
        is_target is False, then index can be whichever source domain was
        passed.

        Returns: (x, y, domain)
        """
        if data is None:
            return None

        assert not is_target or index == 0, \
            "only support one target at present"

        x, y, example_id = data
        d = tf.ones_like(y)*self.domain_label(index=index, is_target=is_target)
        data_target = (x, y, d, example_id)

        return data_target

    # Allow easily overriding each part of the train_step() function, without
    # having to override train_step() in its entirety
    @tf.function
    def prepare_data(self, data_sources, data_target, which_model):
        """ Prepare the data for the model, e.g. by concatenating all sources
        together. Note: do not put code in here that changes the domain labels
        since you presumably want that during evaluation too. Put that in
        domain_label()

        Input xs shape: [num_domains, num_modalities, per-modality x shape...]
        Output xs shape: [num_modalities, per-modality x shape...]

        Note: compile by adding @tf.function decorator if the contents of this
        (overrided) function is compilable
        """
        # By default (e.g. for no adaptation or domain generalization), ignore
        # the target data
        xs_a, y_a, domain_a, example_ids = data_sources

        # Concatenate all source domains' data
        #
        # xs is a list of domains, which is a tuple of modalities, which is
        # tensors, e.g. if one modality but two sources:
        #     [(s1 tensor,), (s2 tensor,)]
        # We want to concatenate the tensors from all domains separately for
        # each modality.
        source_num_modalities, target_num_modalities = self.get_num_modalities()
        xs = [
            tf.concat([x_a[i] for x_a in xs_a], axis=0)
            for i in range(source_num_modalities)
        ]

        task_y_true = tf.concat(y_a, axis=0)
        domain_y_true = tf.concat(domain_a, axis=0)
        auxiliary_data = None

        return xs, task_y_true, domain_y_true, auxiliary_data

    def prepare_data_eval(self, data, is_target):
        """ Prepare the data for the model, e.g. by concatenating all sources
        together. This is like prepare_data() but use during evaluation.

        Input xs shape: [num_domains, num_modalities, per-modality x shape...]
        Output xs shape: [num_modalities, per-modality x shape...]
        """
        xs, y, domain, example_ids = data

        for x in xs:
            assert isinstance(x, list), \
                "Must pass xs=[[...],[...],...] even if only one domain for tf.function consistency"
        assert isinstance(y, list), \
            "Must pass y=[...] even if only one domain for tf.function consistency"
        assert isinstance(domain, list), \
            "Must pass domain=[...] even if only one domain for tf.function consistency"

        # Concatenate all the data (e.g. if multiple source domains)
        source_num_modalities, target_num_modalities = self.get_num_modalities()
        num_modalities = target_num_modalities if is_target else source_num_modalities
        xs = [
            tf.concat([x[i] for x in xs], axis=0)
            for i in range(num_modalities)
        ]
        y = tf.concat(y, axis=0)
        domain = tf.concat(domain, axis=0)
        auxiliary_data = None

        return xs, y, domain, auxiliary_data

    def post_data_eval(self, task_y_true, task_y_pred, domain_y_true,
            domain_y_pred):
        """ Optionally do something with the data after feeding through the
        model. Since the model outputs logits, here we actually take the softmax
        so that during evaluation we have probability distributions. """
        task_y_pred = tf.nn.softmax(task_y_pred)
        domain_y_pred = [tf.nn.softmax(d) for d in domain_y_pred]
        return task_y_true, task_y_pred, domain_y_true, domain_y_pred

    def call_model(self, xs, which_model, is_target=None, **kwargs):
        return self.model[which_model](xs, **kwargs)

    def compute_losses(self, xs, task_y_true, domain_y_true, task_y_pred,
            domain_y_pred, fe_output, contrastive_output, auxiliary_data,
            which_model, training):
        # Maybe: regularization = sum(model.losses) and add to loss
        return self.task_loss(task_y_true, task_y_pred)

    def compute_gradients(self, tape, loss, which_model):
        return tape.gradient(loss,
            self.model[which_model].trainable_variables_task_fe)

    def apply_gradients(self, grad, which_model):
        self.opt[which_model]["opt"].apply_gradients(zip(grad,
            self.model[which_model].trainable_variables_task_fe))

    def train_step(self):
        """
        Get batch of data, prepare data, run through model, compute losses,
        apply the gradients

        Override the individual parts with prepare_data(), call_model(),
        compute_losses(), compute_gradients(), and apply_gradients()

        We return the batch of data so we can use the exact same training batch
        for the "train" evaluation metrics.
        """
        # TensorFlow errors constructing the graph (with tf.function, which
        # makes training faster) if we don't know the data size. Thus, first
        # load batches, then pass to compiled train step.
        all_data_sources = []
        all_data_target = []

        for i in range(self.ensemble_size):
            data_sources, data_target = self.get_next_train_data()
            all_data_sources.append(data_sources)
            all_data_target.append(data_target)

            # If desired, use the same batch for each of the models.
            if FLAGS.ensemble_same_data:
                break

        for i in range(self.ensemble_size):
            # Get random batch for this model in the ensemble (either same for
            # all or different for each)
            if FLAGS.ensemble_same_data:
                data_sources = all_data_sources[0]
                data_target = all_data_target[0]
            else:
                data_sources = all_data_sources[i]
                data_target = all_data_target[i]

            # Prepare
            xs, task_y_true, domain_y_true, auxiliary_data = self.prepare_data(
                data_sources, data_target, which_model=i
            )

            # We compile the entire model call, loss computation, and gradient
            # update/apply
            self._update_model(
                xs, task_y_true, domain_y_true, auxiliary_data, which_model=i
            )

        # We return the first one since we don't really care about the "train"
        # evaluation metrics that much.
        return all_data_sources[0], all_data_target[0]

    @tf.function
    def _update_model(self, xs, task_y_true, domain_y_true, auxiliary_data,
            which_model):
        # Run batch through the model and compute loss
        with tf.GradientTape(persistent=True) as tape:
            task_y_pred, domain_y_pred, fe_output, contrastive_output = \
                self.call_model(xs, which_model=which_model, training=True)
            losses = self.compute_losses(xs, task_y_true, domain_y_true,
                task_y_pred, domain_y_pred, fe_output, contrastive_output,
                auxiliary_data, which_model=which_model, training=True)

        # Update model
        gradients = self.compute_gradients(tape, losses, which_model=which_model)
        del tape
        self.apply_gradients(gradients, which_model=which_model)

    def eval_step(self, data, is_target, is_single_domain):
        """ Evaluate a batch of source or target data, called in metrics.py.
        This preprocesses the data to have x, y, domain always be lists so
        we can use the same compiled tf.function code in eval_step_list() for
        both sources and target domains. """
        xs, y, domain, example_ids = data

        # If it's a single domain,to make later code consistent, embed in a
        # single-item list.
        if is_single_domain:
            xs = [xs]

        # Convert any tuples to lists
        xs_list = []
        for x in xs:
            if isinstance(x, tuple):
                # convert the tuple from load_datasets.py (avoiding a TF error)
                # back to a list
                x = list(x)
            xs_list.append(x)
        xs = xs_list

        if not isinstance(y, list):
            y = [y]
        if not isinstance(domain, list):
            domain = [domain]
        if not isinstance(example_ids, list):
            example_ids = [example_ids]

        return self.eval_step_list((xs, y, domain, example_ids), is_target)

    def add_multiple_losses(self, losses, average=False):
        """
        losses = [
            [total_loss1, task_loss1, ...],
            [total_loss2, task_loss2, ...],
            ...
        ]

        returns [total_loss, task_loss, ...] either the sum or average
        """
        losses_added = None

        for loss_list in losses:
            # If no losses yet, then just set to this
            if losses_added is None:
                losses_added = loss_list
            # Otherwise, add to the previous loss values
            else:
                assert len(losses_added) == len(loss_list), \
                    "subsequent losses have different length than the first"

                for i, loss in enumerate(loss_list):
                    losses_added[i] += loss

        assert losses_added is not None, \
            "must return losses from at least one domain"

        if average:
            averaged_losses = []

            for loss in losses_added:
                averaged_losses.append(loss / len(losses))

            return averaged_losses
        else:
            return losses_added

    #@tf.function  # faster not to compile
    def eval_step_list(self, data, is_target):
        """ Override preparation in prepare_data_eval() """
        (
            xs,
            orig_task_y_true,
            orig_domain_y_true,
            auxiliary_data,
        ) = self.prepare_data_eval(data, is_target)

        task_y_true_list = []
        task_y_pred_list = []
        domain_y_true_list = []
        domain_y_pred_list = []
        losses_list = []

        for i in range(self.ensemble_size):
            # Run through model
            #
            # We don't need to handle both_domains_simultaneously here because
            # during evaluation we already pass the source(s) data through
            # first and the target data through separately second.
            task_y_pred, domain_y_pred, fe_output, contrastive_output = \
                self.call_model(xs, which_model=i, is_target=is_target,
                training=False)

            # Calculate losses
            losses = self.compute_losses(xs, orig_task_y_true,
                orig_domain_y_true, task_y_pred, domain_y_pred, fe_output,
                contrastive_output, auxiliary_data, which_model=i,
                training=False)

            if not isinstance(losses, list):
                losses = [losses]

            losses_list.append(losses)

            # Post-process data (e.g. compute softmax from logits)
            task_y_true, task_y_pred, domain_y_true, domain_y_pred = \
                self.post_data_eval(orig_task_y_true, task_y_pred,
                    orig_domain_y_true, domain_y_pred)

            task_y_true_list.append(task_y_true)
            task_y_pred_list.append(task_y_pred)
            domain_y_true_list.append(domain_y_true)
            domain_y_pred_list += domain_y_pred  # list from multiple modalities

        # Combine information from each model in the ensemble -- averaging.
        #
        # Note: this is how the ensemble predictions are made with InceptionTime
        # having an ensemble of 5 models -- they average the softmax outputs
        # over the ensemble (and we now have softmax after the post_data_eval()
        # call). See their code:
        # https://github.com/hfawaz/InceptionTime/blob/master/classifiers/nne.py
        task_y_true_avg = tf.math.reduce_mean(task_y_true_list, axis=0)
        task_y_pred_avg = tf.math.reduce_mean(task_y_pred_list, axis=0)
        domain_y_true_avg = tf.math.reduce_mean(domain_y_true_list, axis=0)
        domain_y_pred_avg = tf.math.reduce_mean(domain_y_pred_list, axis=0)
        losses_avg = self.add_multiple_losses(losses_list, average=True)

        return task_y_true_avg, task_y_pred_avg, domain_y_true_avg, \
            domain_y_pred_avg, losses_avg


# The base method class performs no adaptation. Use this by passing the "target"
# as the source and not specifying a target domain. Then, set shared_modalities
# to be "0" to train just on modality 0, or "0,1" for both, or "1" for just
# modality 1, etc.
@register_method("none")
class MethodNone(MethodBase):
    def create_model(self, model_name):
        # We need to create the right number of FE's and DC's otherwise the
        # closed method will error, though the open/partial will work (assuming
        # there's only 2 modalities, since then we only have 1 shared)
        num_feature_extractors, num_domain_classifiers \
            = self.calc_num_components()

        return models.BasicModel(self.num_classes, self.domain_outputs,
            model_name=model_name,
            num_feature_extractors=num_feature_extractors,
            num_domain_classifiers=num_domain_classifiers,
            shared_modalities=self.shared_modalities,
            share_most_weights=self.share_most_weights)

    def _keep_shared(self, xs):
        """ Keep only the domains set in --shared_modalities and the order """
        return [xs[modality] for modality in self.shared_modalities]

    @tf.function
    def prepare_data(self, data_sources, data_target, which_model):
        xs, task_y_true, domain_y_true, auxiliary_data = super().prepare_data(
            data_sources, data_target, which_model
        )
        return self._keep_shared(xs), task_y_true, domain_y_true, auxiliary_data

    def prepare_data_eval(self, data, is_target):
        xs, y, domain, auxiliary_data = super().prepare_data_eval(data, is_target)
        return self._keep_shared(xs), y, domain, auxiliary_data


@register_method("codats")
class MethodDann(MethodBase):
    """
    DANN / CoDATS (with modal architecture and MS-DA, then CoDATS)

    Only supports one modality. We create only one feature extractor in
    create_model() and drop the other modalities in prepare_data() and
    prepare_data_eval(), only including the one modality specified
    in --shared_modalities=...
    """
    def __init__(self, source_datasets, target_dataset,
            global_step, total_steps, *args, **kwargs):
        self.global_step = global_step  # should be TF variable
        self.total_steps = total_steps
        super().__init__(source_datasets, target_dataset, *args, **kwargs)
        self.loss_names += ["task", "domain"]

    def create_model(self, model_name):
        # By default we create only one FE, so make sure we only have one
        # shared modality. We drop the non-shared modalities in prepare_data()
        # and prepare_data_eval().
        assert len(self.shared_modalities) == 1, \
            "DANN only supports one shared modality between domains"

        return models.DannModel(self.num_classes, self.domain_outputs,
            self.global_step, self.total_steps, model_name=model_name)

    def create_optimizers(self):
        opt = super().create_optimizers()
        # We need an additional optimizer for DANN
        opt["d_opt"] = self.create_optimizer(
            learning_rate=FLAGS.lr*FLAGS.lr_domain_mult)
        return opt

    def create_losses(self):
        # Note: at the moment these are the same, but if we go back to
        # single-source, then the domain classifier may be sigmoid not softmax
        super().create_losses()
        self.domain_loss = make_loss()

    @tf.function
    def prepare_data(self, data_sources, data_target, which_model):
        assert data_target is not None, "cannot run DANN without target"
        xs_a, y_a, domain_a, example_ids_a = data_sources
        xs_b, y_b, domain_b, example_ids_b = data_target

        # Concatenate all source domains' data
        #
        # xs is a list of domains, which is a tuple of modalities, which is
        # tensors, e.g. if one modality but two sources:
        #     [(s1 tensor,), (s2 tensor,)]
        # We want to concatenate the tensors from all domains separately for
        # each modality.
        source_num_modalities, target_num_modalities = self.get_num_modalities()
        xs_a = [
            tf.concat([x_a[i] for x_a in xs_a], axis=0)
            for i in range(source_num_modalities)
        ]
        y_a = tf.concat(y_a, axis=0)
        domain_a = tf.concat(domain_a, axis=0)

        # Concatenate for adaptation - concatenate source labels with all-zero
        # labels for target since we can't use the target labels during
        # unsupervised domain adaptation
        #
        # We concatenate the shared modalities, dropping the non-shared
        # modalities. DANN by itself doesn't know how to handle changes in the
        # number of modalities between domains.
        xs = [
            tf.concat((xs_a[modality], xs_b[modality]), axis=0)
            for modality in self.shared_modalities
        ]
        task_y_true = tf.concat((y_a, tf.zeros_like(y_b)), axis=0)
        domain_y_true = tf.concat((domain_a, domain_b), axis=0)
        auxiliary_data = None

        return xs, task_y_true, domain_y_true, auxiliary_data

    def prepare_data_eval(self, data, is_target):
        """ Prepare the data for the model, e.g. by concatenating all sources
        together. This is like prepare_data() but use during evaluation. """
        xs, y, domain, auxiliary_data = super().prepare_data_eval(data, is_target)
        xs = [
            xs[modality] for modality in self.shared_modalities
        ]

        return xs, y, domain, auxiliary_data

    def compute_losses(self, xs, task_y_true, domain_y_true, task_y_pred,
            domain_y_pred, fe_output, contrastive_output, auxiliary_data,
            which_model, training):
        nontarget = tf.where(tf.not_equal(domain_y_true, 0))
        task_y_true = tf.gather(task_y_true, nontarget, axis=0)
        task_y_pred = tf.gather(task_y_pred, nontarget, axis=0)

        task_loss = self.task_loss(task_y_true, task_y_pred)
        d_loss = sum([
            self.domain_loss(domain_y_true, d)
            for d in domain_y_pred
        ])/len(domain_y_pred)
        total_loss = task_loss + d_loss
        return [total_loss, task_loss, d_loss]

    def compute_gradients(self, tape, losses, which_model):
        total_loss, task_loss, d_loss = losses
        grad = tape.gradient(total_loss,
            self.model[which_model].trainable_variables_task_fe_domain)
        d_grad = tape.gradient(d_loss,
            self.model[which_model].trainable_variables_domain)
        return [grad, d_grad]

    def apply_gradients(self, gradients, which_model):
        grad, d_grad = gradients
        self.opt[which_model]["opt"].apply_gradients(zip(grad,
            self.model[which_model].trainable_variables_task_fe_domain))
        # Update discriminator again
        self.opt[which_model]["d_opt"].apply_gradients(zip(d_grad,
            self.model[which_model].trainable_variables_domain))


class MethodDawsBase(MethodDann):
    """ Domain adaptation with weak supervision (in this case, target-domain
    label proportions)"""
    def __init__(self, *args, noise_amount=None, **kwargs):
        super().__init__(*args, **kwargs)

        if noise_amount is None:
            noise_amount = FLAGS.ws_noise

        self.loss_names += ["weak"]
        self.compute_p_y(noise_amount)

    def compute_p_y(self, noise_amount=None):
        """ Compute P(y) (i.e. class balance) of the training target dataset

        Note: we simulate the self-report label proportions from looking at
        the target training labels (not validation or test sets). However, after
        this function call, we don't use the labels themselves (outside of
        computing evaluation accuracy), just the computed proportions for the
        training.
        """
        # Compute proportion of each class
        # Note: we use the "eval" train dataset since it doesn't repeat infinitely
        # and we use "train" not test since we don't want to peak at the
        # validation data we use for model selection.
        self.p_y = class_balance(self.target_train_eval_dataset, self.num_classes)

        print("Correct proportions:", self.p_y)
        before = self.p_y * 1.0
        total_noise = 0

        # Add noise
        if noise_amount is not None and noise_amount != 0:
            # Make sure it's float
            noise_amount = float(noise_amount)

            while noise_amount > 0:
                # Randomly pick a class
                label = random.randint(0, len(self.p_y)-1)

                # Randomly pick some amount of our noise budget. Always positive
                # since otherwise we could do this loop forever.
                noise = random.uniform(0.000001, noise_amount)

                # We skip negatives since afterwards we normalize, so adding
                # to other classes results in a negative.
                #
                # Randomly pick whether to add/subtract this from the class
                # proportions (otherwise it's always additive noise)
                # positive = random.choice([True, False])
                # if not positive:
                #     noise *= -1

                # Update the class proportions
                new_value = self.p_y[label] + noise

                # Noise was too much, so break before we add it and go over.
                # if noise_amount - noise < 0:
                #     break

                # Skip and try again if we end up with a negative or zero
                # proportion. Also, make sure we don't add too much noise.
                if new_value > 0:
                    print("{} {} {} label {}".format(
                        "Adding" if noise > 0 else "Subtracting",
                        noise,
                        "to" if noise > 0 else "from",
                        label,
                    ))
                    self.p_y[label] = new_value

                    # Subtract the amount we used from the noise budget
                    noise_amount -= noise
                    # Debugging
                    total_noise += noise

            print("Noised proportions:", self.p_y)
            print("Sum difference:", sum([abs(self.p_y[i]-before[i]) for i in range(len(self.p_y))]))
            print("Total noise:", total_noise)

            # Re-normalize so the sum still is 1
            self.p_y = self.p_y / sum(self.p_y)
            print("Normalized noised proportions:", self.p_y)
            print("Sum difference norm:", sum([abs(self.p_y[i]-before[i]) for i in range(len(self.p_y))]))

    def compute_losses(self, xs, task_y_true, domain_y_true, task_y_pred,
            domain_y_pred, fe_output, contrastive_output, auxiliary_data,
            which_model, training):
        # DANN losses
        nontarget = tf.where(tf.not_equal(domain_y_true, 0))
        task_y_true_nontarget = tf.gather(task_y_true, nontarget, axis=0)
        task_y_pred_nontarget = tf.gather(task_y_pred, nontarget, axis=0)

        task_loss = self.task_loss(task_y_true_nontarget, task_y_pred_nontarget)
        d_loss = sum([
            self.domain_loss(domain_y_true, d)
            for d in domain_y_pred
        ])/len(domain_y_pred)

        # DA-WS regularizer
        #
        # Get predicted target-domain labels. We ignore label proportions for
        # the source domains since we train to predict the correct labels there.
        # We don't know the target-domain labels, so instead we try using this
        # additional P(y) label proportion information. Thus, we use it and the
        # adversarial domain-invariant FE objectives as sort of auxiliary
        # losses.
        target = tf.where(tf.equal(domain_y_true, 0))
        task_y_pred_target = tf.gather(task_y_pred, target, axis=0)

        # Idea:
        # argmax, one-hot, reduce_sum(..., axis=1), /= batch_size, KL with p_y
        # However, argmax yields essentially useless gradients (as far as I
        # understand it, e.g. we use cross entropy loss for classification not
        # the actual 0-1 loss or loss on the argmax of the softmax outputs)
        #
        # Thus, a soft version. Idea: softmax each, reduce sum vertically,
        #   /= batch_size, then KL
        # This is different than per-example-in-batch KLD because we average
        # over the softmax outputs across the batch before KLD. So, the
        # difference is whether averaging before or after KLD.
        #
        # Note: this depends on a large enough batch size. If you can't set it
        # >=64 or so (like what we use in SS-DA for the target data, i.e. half
        # the 128 batch size), then accumulate this gradient over multiple steps
        # and then apply.
        #
        # cast batch_size to float otherwise:
        # "x and y must have the same dtype, got tf.float32 != tf.int32"
        batch_size = tf.cast(tf.shape(task_y_pred_target)[0], dtype=tf.float32)
        p_y_batch = tf.reduce_sum(tf.nn.softmax(task_y_pred_target), axis=0) / batch_size
        daws_loss = tf.keras.losses.KLD(self.p_y, p_y_batch)

        # Sum up individual losses for the total
        #
        # Note: daws_loss doesn't have the DANN learning rate schedule because
        # it goes with the task_loss. We want to learn predictions for the task
        # classifier that both correctly predicts labels on the source data and
        # on the target data aligns with the correct label proportions.
        # Separately, we want the FE representation to also be domain invariant,
        # which we apply the learning rate schedule to, I think, to help the
        # adversarial part converge properly (recall GAN training instability
        # stuff).
        total_loss = task_loss + d_loss + daws_loss

        return [total_loss, task_loss, d_loss, daws_loss]

    def compute_gradients(self, tape, losses, which_model):
        # We only use daws_loss for plotting -- for computing gradients it's
        # included in the total loss
        return super().compute_gradients(tape, losses[:-1], which_model)


@register_method("codats_ws")
class MethodCodatsWS(MethodDawsBase):
    pass


#
# Contrastive Adversarial Learning for Multi-Source Time Series Domain Adaptation
#
class MethodCaldaBase:
    """ CALDA - instantiations chosen via arguments

    For Domain Adaptation, also inherit from MethodDann.
    For Domain Generalization, also inherit from MethodDannDG.
    """
    def __init__(self, *args, num_contrastive_units=None,
            pseudo_label_target=False, hard=False,
            in_domain=False, any_domain=False,
            domain_generalization=False, weight_adversary=1, **kwargs):
        self.weight_adversary = weight_adversary
        self.weight_similarity = FLAGS.similarity_weight
        self.pseudo_label_target = pseudo_label_target
        self.hard = hard
        self.in_domain = in_domain
        self.any_domain = any_domain
        self.domain_generalization = domain_generalization

        if num_contrastive_units is None:
            self.num_contrastive_units = FLAGS.contrastive_units
        else:
            self.num_contrastive_units = num_contrastive_units

        super().__init__(*args, **kwargs)
        self.loss_names += ["similarity"]

    def create_losses(self):
        super().create_losses()
        # Used in non-compiled version
        self.cl_crossentropy_loss = make_loss()
        # Used in hard pos/neg
        self.task_no_reduction_loss = make_loss(
            reduction=tf.keras.losses.Reduction.NONE)

    def create_model(self, model_name):
        # Not multi-modal at the moment
        assert len(self.shared_modalities) == 1, \
            "only supports one shared modality between domains"

        return models.DannModel(self.num_classes, self.domain_outputs,
            self.global_step, self.total_steps,
            num_contrastive_units=self.num_contrastive_units,
            model_name=model_name)

    def _cartesian_product(self, a, b):
        """
        Assumes a and b are 1D
        https://stackoverflow.com/a/47133461
        """
        tile_a = tf.tile(tf.expand_dims(a, 1), [1, tf.shape(b)[0]])
        tile_a = tf.expand_dims(tile_a, 2)
        tile_b = tf.tile(tf.expand_dims(b, 0), [tf.shape(a)[0], 1])
        tile_b = tf.expand_dims(tile_b, 2)
        cartesian_product = tf.concat([tile_a, tile_b], axis=2)
        return cartesian_product

    def _contrastive_get_examples(self, task_y_true, domain_y_true,
            z_output, task_y_pred):
        """
        Get y, d, z, and pred used in contrastive learning

        Take the true domain and class labels and model outputs and get the
        source-only or if pseudo labeling the source + pseudo-labeled target
        """
        # For domain generalization, d=0 is the first source domain not the
        # target domain since in domain generalization we don't have target
        # domain data. Thus, just use everything, but do range here so we don't
        # have to rewrite the rest of the code.
        if self.domain_generalization:
            nontarget = tf.expand_dims(tf.range(0, tf.shape(domain_y_true)[0]), axis=-1)
        else:
            # Anchors = the full batch, excluding the target data
            nontarget = tf.where(tf.not_equal(domain_y_true, 0))

        y = tf.gather(task_y_true, nontarget, axis=0)
        d = tf.gather(domain_y_true, nontarget, axis=0)
        z = tf.gather(z_output, nontarget, axis=0)
        pred = tf.gather(task_y_pred, nontarget, axis=0)

        # If we include the target domain, since we don't know the true labels,
        # we instead take the current task classifier predictions for the target
        # data as the true labels, i.e. pseudo labeling the target data. Then
        # concatenate with the non-target y/d/z we generated above.
        if self.pseudo_label_target:
            assert not self.domain_generalization, \
                "there's no target data to pseudo label in CALDG"

            # Anchors = the full batch, excluding the target data
            target = tf.where(tf.equal(domain_y_true, 0))

            # Get y from task_y_pred rather than task_y_true
            target_y = tf.gather(task_y_pred, target, axis=0)
            # The true y's are sparse, so get class label integer from the
            # raw logits
            target_y = tf.cast(
                tf.argmax(tf.nn.softmax(target_y), axis=-1), dtype=tf.float32)
            # d and z as usual
            target_d = tf.gather(domain_y_true, target, axis=0)
            target_z = tf.gather(z_output, target, axis=0)
            target_pred = tf.gather(task_y_pred, target, axis=0)

            # Concatenate everything
            y = tf.concat([y, target_y], axis=0)
            d = tf.concat([d, target_d], axis=0)
            z = tf.concat([z, target_z], axis=0)
            pred = tf.concat([pred, target_pred], axis=0)

        # Get rid of extra "1" dimensions, otherwise these are of shapes
        # [num_total, 1], [num_total, 1], and
        # [num_total, 1, contrastive_units] respectively.
        #
        # Note: wrote code/examples without this extra dimension, so removing
        # to avoid rewriting everything.
        y = tf.squeeze(y, axis=1)
        d = tf.squeeze(d, axis=1)
        z = tf.squeeze(z, axis=1)
        pred = tf.squeeze(pred, axis=1)

        # Compute after we possibly append the target data
        num_total = tf.shape(y)[0]

        return y, d, z, pred, num_total

    def _contrastive_get_indices(self, y, d, anchors, num_total):
        """
        Get the anchor, positives, and negatives indices into the y/d arrays
        based on whether we want in/any/cross domain pairs for contrastive
        learning
        """
        # Positive/negative masks
        prod_indices = self._cartesian_product(anchors, tf.range(0, num_total))
        # Note while these are called "all", if no pseudo-labeling then they
        # still exclude the target domain...
        anchor_d = tf.gather_nd(d, tf.expand_dims(prod_indices[:, :, 0], axis=-1))
        all_d = tf.gather_nd(d, tf.expand_dims(prod_indices[:, :, 1], axis=-1))
        anchor_y = tf.gather_nd(y, tf.expand_dims(prod_indices[:, :, 0], axis=-1))
        all_y = tf.gather_nd(y, tf.expand_dims(prod_indices[:, :, 1], axis=-1))
        # Whether the column corresponds to the anchor
        is_not_anchor = tf.logical_not(
            tf.cast(
                tf.gather_nd(tf.eye(num_total), tf.expand_dims(anchors, axis=-1)),
                dtype=tf.bool
            )
        )

        # Within-source label-contrastive learning
        #
        # Note: we need to exclude the anchor in the positives in this case. Not
        # the negatives though since it has a different label so is guaranteed
        # to not be the anchor.
        if self.in_domain:
            # Same domain, same label, not anchor
            positives = tf.logical_and(
                tf.logical_and(tf.equal(anchor_d, all_d), tf.equal(anchor_y, all_y)),
                is_not_anchor
            )
            # Same domain, different label
            negatives = tf.logical_and(
                tf.equal(anchor_d, all_d), tf.not_equal(anchor_y, all_y)
            )
        # Label-contrastive learning (all sources grouped/lumped together)
        elif self.any_domain:
            # Any domain, same label, not anchor
            positives = tf.logical_and(tf.equal(anchor_y, all_y), is_not_anchor)
            # Any domain, different label
            negatives = tf.not_equal(anchor_y, all_y)
        # Cross-source label-contrastive learning
        else:
            # Different domain, same label
            positives = tf.logical_and(
                tf.not_equal(anchor_d, all_d), tf.equal(anchor_y, all_y)
            )
            # Different domain, different label
            negatives = tf.logical_and(
                tf.not_equal(anchor_d, all_d), tf.not_equal(anchor_y, all_y)
            )

        # Positive/negative indices (w.r.t. z)
        #
        # Get indices of values with non-zero in mask, i.e. the real
        # positives/negatives. The tf.where returns the index into the mask
        # which since we take a subset isn't the index into z.
        positives_similarity_indices = tf.gather_nd(
            prod_indices, tf.where(tf.not_equal(tf.cast(positives, tf.int32), 0))
        )
        negatives_similarity_indices = tf.gather_nd(
            prod_indices, tf.where(tf.not_equal(tf.cast(negatives, tf.int32), 0))
        )

        # Also get the anchor indices
        anchors_indices = positives_similarity_indices[:, 0]

        return (
            anchors_indices,
            positives_similarity_indices,
            negatives_similarity_indices,
        )

    def _contrastive_sampling(self, y, pred, anchors_indices,
            positives_similarity_indices, negatives_similarity_indices,
            num_anchors):
        """
        Subset of positives/negatives - random sampling or hard sampling
        """
        # These won't hold for each anchor individually like the non-vectorized
        # version. Instead, it's easier (and maybe better anyway?) to on average
        # have no more than max_positives positives, etc.
        if FLAGS.max_positives != 0:
            max_total_positives = num_anchors * FLAGS.max_positives
        else:
            max_total_positives = None
        if FLAGS.max_negatives != 0:
            max_total_negatives = num_anchors * FLAGS.max_negatives
        else:
            max_total_negatives = None

        # If desired, pick the hard positives/negatives (in terms of task loss)
        # rather than randomly
        if self.hard:
            assert FLAGS.max_positives != 0 or FLAGS.max_negatives != 0, \
                "hard only makes sense if there's a limit on the number" \
                "of positives or negatives. Otherwise, all of them are used, so" \
                "sorting by hardness doesn't change anything."

            # Same as normal task loss, but this doesn't average over the batch.
            # Thus, we can get the per-example loss. Shape: [batch_size]
            hardness = self.task_no_reduction_loss(y, pred)

            # Get hardness of positives (note [:,0] is the anchor)
            pos_hardness = tf.gather(hardness, positives_similarity_indices[:, 1])

            # Indices ordered by hardness, with max pos/neg
            pos_hardsort = tf.argsort(pos_hardness, axis=-1, direction='DESCENDING', stable=False)

            if FLAGS.max_positives != 0:
                pos_hardsort = pos_hardsort[:max_total_positives]

            # Get the anchors/positives
            positives_similarity_indices = tf.gather(positives_similarity_indices, pos_hardsort)
            anchors_indices = tf.gather(anchors_indices, pos_hardsort)

            # How we define "hard" for negatives
            #
            # There's only one way for a positive to be right (disregarding
            # uncertainty of the softmax output). Thus, if the task loss is
            # high then the representation is (probably) different when it
            # should be similar.
            #
            # However, there's many ways to be wrong for the negatives. For
            # example, if the anchor's label is 0, one negative may have
            # high task loss if it should be label 1 but was 0 or if it
            # should be label 1 but was 2. The first case is primarily what
            # we're interested in: push apart the negatives that are
            # currently similar but should really be different.
            #
            # We do this via:
            # - Select the negatives
            # - Compute task loss, where true = anchor's label
            # - Take the *lowest* loss (rather than highest before): the
            #   examples that the classifier (incorrectly) most thinks are
            #   the same class as the anchor (probably currently similar in
            #   the representation space). We'll push those apart.

            # Select negatives
            negative_anchor_labels = tf.gather_nd(y, tf.expand_dims(negatives_similarity_indices[:, 0], axis=-1))
            negative_predicted_labels = tf.gather_nd(pred, tf.expand_dims(negatives_similarity_indices[:, 1], axis=-1))
            # Compute task loss - note the negative doesn't actually have
            # the anchor's label ("true" label) here
            neg_hardness = self.task_no_reduction_loss(negative_anchor_labels, negative_predicted_labels)
            # Sort ascending (not descending like for the positives)
            neg_hardsort = tf.argsort(neg_hardness, axis=-1, direction='ASCENDING', stable=False)

            # Subset negatives from the above definition of "hard"
            if FLAGS.max_negatives != 0:
                neg_hardsort = neg_hardsort[:max_total_negatives]

            negatives_similarity_indices = tf.gather(negatives_similarity_indices, neg_hardsort)
        else:
            if FLAGS.max_positives != 0:
                # Need to shuffle both the anchor indices and positive similarity
                # indices together, so shuffle the rows/indices then gather those to get
                # the shuffled results.
                shuffled_subset = tf.random.shuffle(
                    tf.range(0, tf.shape(positives_similarity_indices)[0])
                )[:max_total_positives]
                positives_similarity_indices = tf.gather(positives_similarity_indices, shuffled_subset)
                anchors_indices = tf.gather(anchors_indices, shuffled_subset)
                tf.assert_equal(tf.shape(anchors_indices)[0], tf.shape(positives_similarity_indices)[0])

            if FLAGS.max_negatives != 0:
                negatives_similarity_indices = tf.random.shuffle(negatives_similarity_indices)[:max_total_negatives]

        # Compute number of positives after we take the subset
        #
        # Note: includes dependency on number of anchors so we will only
        # normalize by this, not by num_anchors as well
        num_positives = tf.shape(positives_similarity_indices)[0]
        # num_negatives = tf.shape(negatives_similarity_indices)[0]
        # tf.print("Num positives", num_positives)
        # tf.print("Num negatives", num_negatives)

        return (
            anchors_indices,
            positives_similarity_indices,
            negatives_similarity_indices,
            num_positives,
        )

    def _cosine_similarity_from_indices(self, vectors, indices):
        """
        Compute cosine similarity between the desired pairs of
        (anchor, positive/negative) z vectors.
        """
        # Using positives as an example...
        #
        # For each (anchor, positive) index pair, replace the index with the
        # actual z value. We get a tensor of shape
        # [num_positives, 2, contrastive_units]
        vectors = tf.gather_nd(vectors, tf.expand_dims(indices, axis=-1))

        # Normalize, i.e. cosine similarity
        vectors = tf.math.l2_normalize(vectors, axis=-1)

        # Compute cosine similarity across pairs. Multiply across the
        # second dimension (the pairs), sum across the components (last
        # dimension).
        return tf.reduce_sum(tf.reduce_prod(vectors, axis=1), axis=-1)

    def _contrastive_infonce(self, positives_similarity, negatives_similarity,
            anchors_indices, positives_similarity_indices,
            negatives_similarity_indices,
            num_total, num_positives, training, debug=False):
        # Temperature
        tau = FLAGS.temperature

        if debug:
            tf.print("negatives_similarity_indices", negatives_similarity_indices.shape)
            tf.print("negatives_similarity", negatives_similarity.shape)
            tf.print("num_total", num_total)
            tf.print("positives_similarity_indices", positives_similarity_indices.shape)
            tf.print("positives_similarity", positives_similarity.shape)

        #
        # Cross entropy loss
        #
        # Put back into the full matrix (since the indices are referenced to
        # z) and then take the subset of anchors we're looking at.
        #
        # Also generate mask to use in cross entropy loss since it's possible
        # the cosine similarity is actually zero, so we can't just check if = 0.
        negatives_square = tf.scatter_nd(negatives_similarity_indices,
            negatives_similarity, [num_total, num_total])
        negatives_square_mask = tf.scatter_nd(negatives_similarity_indices,
            tf.ones_like(negatives_similarity), [num_total, num_total])
        negatives_for_anchors = tf.gather(negatives_square, anchors_indices)
        negatives_for_anchors_mask = tf.gather(negatives_square_mask, anchors_indices)

        if debug:
            tf.print("anchors_indices", anchors_indices.shape, anchors_indices)
            tf.print("negatives_square", negatives_square.shape)
            tf.print("negatives_for_anchors", negatives_for_anchors.shape)

        # We're not using the built-in cross entropy loss since we have to
        # handle the mask. Thus, we don't actually need to concatenate since
        # we know the true/correct value was positives_similarity.
        ce_positive = tf.math.exp(positives_similarity/tau)
        # Negatives with the mask, otherwise e^0 = 1, i.e. affects loss
        ce_negatives = tf.multiply(tf.math.exp(negatives_for_anchors/tau), negatives_for_anchors_mask)

        if debug:
            tf.print("Positives:", ce_positive.shape)
            tf.print("Negatives:", ce_negatives.shape)
            tf.print("Negatives reduce sum:", tf.reduce_sum(ce_negatives, axis=-1).shape)

        denominator = ce_positive + tf.reduce_sum(ce_negatives, axis=-1)
        cross_entropy_loss_from_logits = - tf.math.log(ce_positive / denominator)

        # Sum over all anchors
        similarity_loss = tf.reduce_sum(cross_entropy_loss_from_logits)

        # Normalize
        similarity_loss = similarity_loss / tf.cast(num_positives, tf.float32)

        return similarity_loss

    def _contrastive_loss(self, task_y_true, domain_y_true,
            z_output, task_y_pred, auxiliary_data, training):
        """ Cross-source similarity loss, based on InfoNCE

        For pseudo code and Python examples, see contrastive_loss.md
        """
        # Get the label, domain, embedding, etc. arrays we'll use. We use only
        # the source domains unless pseudo labeling, in which case we add the
        # pseudo-labeled target domain data as well.
        y, d, z, pred, num_total = self._contrastive_get_examples(task_y_true,
            domain_y_true, z_output, task_y_pred)

        # Since we have rearranged y/d/z and they no longer (necessarily)
        # correspond to the data in task_y_true, domain_y_true, ... we can just
        # make the anchors the indices into the entries of y/d/z
        anchors = tf.range(0, num_total)
        num_anchors = tf.shape(anchors)[0]

        # Limit number of anchors so this doesn't take forever, since
        # O(num_anchors^2). However, don't limit the positives/negatives to just
        # the anchor examples. Those can come from anywhere in the minibatch.
        if FLAGS.max_anchors != 0:
            anchors = tf.random.shuffle(anchors)[:FLAGS.max_anchors]

        # Get anchor, positive, and negative indices, which are one of
        # in/any/cross domain
        (
            anchors_indices,
            positives_similarity_indices,
            negatives_similarity_indices,
        ) = self._contrastive_get_indices(y, d, anchors, num_total)

        # Subset the anchors, positives, and negatives either randomly or hard
        # sampling
        (
            anchors_indices,
            positives_similarity_indices,
            negatives_similarity_indices,
            num_positives,
        ) = self._contrastive_sampling(y, pred, anchors_indices,
            positives_similarity_indices, negatives_similarity_indices,
            num_anchors)

        # Compute similarity (e.g. cosine similarity) between anchor-positive
        # and anchor-negative pairs
        positives_similarity = self._cosine_similarity_from_indices(
            z, positives_similarity_indices)
        negatives_similarity = self._cosine_similarity_from_indices(
            z, negatives_similarity_indices)

        # Compute the InfoNCE loss
        contrastive_loss = self._contrastive_infonce(
            positives_similarity, negatives_similarity,
            anchors_indices, positives_similarity_indices,
            negatives_similarity_indices,
            num_total, num_positives, training)

        return contrastive_loss

    def compute_losses(self, xs, task_y_true, domain_y_true, task_y_pred,
            domain_y_pred, fe_output, contrastive_output, auxiliary_data,
            which_model, training):
        # Normal task and domain classifier losses
        other_losses = super().compute_losses(
            xs, task_y_true, domain_y_true, task_y_pred,
            domain_y_pred, fe_output, contrastive_output, auxiliary_data,
            which_model, training)

        # No weak supervision
        if len(other_losses) == 3:
            weak_supervision = False
            _, task_loss, d_loss = other_losses
        # Weak supervision
        elif len(other_losses) == 4:
            weak_supervision = True
            _, task_loss, d_loss, daws_loss = other_losses
        else:
            raise NotImplementedError("should be 3 or 4 losses")

        # Additional contrastive loss
        similarity_loss = self._contrastive_loss(
            task_y_true, domain_y_true, contrastive_output, task_y_pred,
            auxiliary_data, training)

        # If weak supervision, include it in the total loss and also return it
        # for plotting in metrics
        if weak_supervision:
            total_loss = task_loss \
                + self.weight_adversary*d_loss \
                + self.weight_similarity*similarity_loss \
                + daws_loss
            return [total_loss, task_loss, d_loss, daws_loss, similarity_loss]
        else:
            total_loss = task_loss \
                + self.weight_adversary*d_loss \
                + self.weight_similarity*similarity_loss
            return [total_loss, task_loss, d_loss, similarity_loss]

    def compute_gradients(self, tape, losses, which_model):
        # similarity loss is included in the total loss, so ignore it here
        return super().compute_gradients(tape, losses[:-1], which_model)


@register_method("calda_xs_r")
class MethodCaldaCrossR(MethodCaldaBase, MethodDann):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)


@register_method("calda_in_r")
class MethodCaldaInR(MethodCaldaBase, MethodDann):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, in_domain=True, **kwargs)


@register_method("calda_any_r")
class MethodCaldaAnyR(MethodCaldaBase, MethodDann):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, any_domain=True, **kwargs)


@register_method("calda_xs_r_p")
class MethodCaldaCrossRP(MethodCaldaBase, MethodDann):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, pseudo_label_target=True, **kwargs)


@register_method("calda_in_r_p")
class MethodCaldaInRP(MethodCaldaBase, MethodDann):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, in_domain=True, pseudo_label_target=True, **kwargs)


@register_method("calda_any_r_p")
class MethodCaldaAnyRP(MethodCaldaBase, MethodDann):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, any_domain=True, pseudo_label_target=True, **kwargs)


@register_method("calda_xs_h")
class MethodCaldaCrossH(MethodCaldaBase, MethodDann):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, hard=True, **kwargs)


@register_method("calda_in_h")
class MethodCaldaInH(MethodCaldaBase, MethodDann):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, hard=True, in_domain=True, **kwargs)


@register_method("calda_any_h")
class MethodCaldaAnyH(MethodCaldaBase, MethodDann):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, hard=True, any_domain=True, **kwargs)


@register_method("calda_xs_h_p")
class MethodCaldaCrossHP(MethodCaldaBase, MethodDann):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, hard=True, pseudo_label_target=True, **kwargs)


@register_method("calda_in_h_p")
class MethodCaldaInHP(MethodCaldaBase, MethodDann):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, hard=True, in_domain=True, pseudo_label_target=True, **kwargs)


@register_method("calda_any_h_p")
class MethodCaldaAnyHP(MethodCaldaBase, MethodDann):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, hard=True, any_domain=True, pseudo_label_target=True, **kwargs)


@register_method("calda_xs_h_ws")
class MethodCaldaCrossHWS(MethodCaldaCrossH, MethodDawsBase):
    pass


@register_method("calda_any_r_ws")
class MethodCaldaAnyRWS(MethodCaldaAnyR, MethodDawsBase):
    pass


@register_method("calda_any_r_noadv")
class MethodCaldaAnyRNoAdv(MethodCaldaBase, MethodDann):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, any_domain=True, weight_adversary=0,
            **kwargs)


@register_method("calda_xs_h_noadv")
class MethodCaldaCrossHNoAdv(MethodCaldaBase, MethodDann):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, hard=True, weight_adversary=0,
            **kwargs)


#
# Domain generalization
#
@register_method("codats_dg")
class MethodDannDG(MethodDann):
    """
    DANN/CoDATS but to make it generalization rather than adaptation:

    - calculate_domain_outputs: don't include a softmax output for the target domain
    - domain_label: don't include target as domain 0, so start sources at 0
    - prepare_data: ignore the target domain when preparing the data for training
    - compute_losses: don't throw out domain 0 data since domain 0 is no longer
      the target
    """
    def calculate_domain_outputs(self):
        # We override it in load_datasets.py now
        # assert FLAGS.batch_division != "all", \
        #     "batch_division all doesn't make sense with DG, use --batch_division=sources"

        # SparseCategoricalCrossentropy gives an error if there's only one class.
        # Thus, throw in an extra, unused class (so softmax output always has 2).
        # Really, why would anybody do DG if there's only one domain...
        #
        # a=tf.constant([[.9, .05, .05], [.5, .89, .6], [.05, .01, .94],
        #   [0.1, 0.6, 0.3]])
        # t=tf.constant([0,1,2,1])
        # cce(t,a)  ## works
        #
        # b=tf.constant([[1.0], [1.0], [1.0], [1.0]])
        # t=tf.constant([0,0,0,0])
        # cce(t,b)  ## errors:
        #   "ValueError: Shape mismatch: The shape of labels (received (342,))
        #   should equal the shape of logits except for the last dimension
        #   (received (1, 4))."
        if self.num_source_domains == 1:
            domain_outputs = 2
        else:
            domain_outputs = self.num_source_domains

        return domain_outputs

    def domain_label(self, index, is_target):
        """
        Shift down the domain labels so 0 is not source 1 since we don't have a
        target domain.

        Note: during evaluation, if target data is used, then the results will
        be wrong since target=0 and source #1=0 for the domain label. However,
        target data shouldn't be used. It may cause some issues in the metrics
        computations though during training (see metrics.py), e.g. the
        target-domain domain classifier accuracy won't make sense.

        New domain labeling:
        0 = target
        0 = source #0
        1 = source #1
        2 = source #2
        ...
        """
        if is_target:
            return 0
        else:
            return index

    @tf.function
    def prepare_data(self, data_sources, data_target, which_model):
        # Ignore target domain data when doing domain generalization
        # (Copied from DANN's but throw out target data)
        xs_a, y_a, domain_a, example_ids_a = data_sources
        xs_b, y_b, domain_b, example_ids_b = data_target

        # Concatenate all source domains' data
        #
        # xs is a list of domains, which is a tuple of modalities, which is
        # tensors, e.g. if one modality but two sources:
        #     [(s1 tensor,), (s2 tensor,)]
        # We want to concatenate the tensors from all domains separately for
        # each modality.
        source_num_modalities, target_num_modalities = self.get_num_modalities()
        xs_a = [
            tf.concat([x_a[i] for x_a in xs_a], axis=0)
            for i in range(source_num_modalities)
        ]
        y_a = tf.concat(y_a, axis=0)
        domain_a = tf.concat(domain_a, axis=0)
        auxiliary_data = None

        return xs_a, y_a, domain_a, auxiliary_data

    def compute_losses(self, xs, task_y_true, domain_y_true, task_y_pred,
            domain_y_pred, fe_output, contrastive_output, auxiliary_data,
            which_model, training):
        # Since we don't have target domain data, don't throw out anything like
        # we did in MethodDANN()
        task_loss = self.task_loss(task_y_true, task_y_pred)
        d_loss = sum([
            self.domain_loss(domain_y_true, d)
            for d in domain_y_pred
        ])/len(domain_y_pred)
        total_loss = task_loss + d_loss
        return [total_loss, task_loss, d_loss]


@register_method("sleep_dg")
class MethodSleepDG(MethodDannDG):
    """ Same as DANN-DG but uses sleep model that feeds task classifier output
    to domain classifier """
    def create_model(self, model_name):
        return models.SleepModel(self.num_classes, self.domain_outputs,
            self.global_step, self.total_steps, model_name=model_name)


@register_method("aflac_dg")
class MethodAflacDG(MethodDannDG):
    """ AFLAC uses KL divergence rather than GRL

    The domain classifier is updated to correctly classify the domain. The task
    classifier to correctly classify the task. However, the feature extractor is
    updated with a combination of making the task classifier correct while also
    wanting the domain classifier's output probabilities to match P(d|y) for
    each known label y.

    For example, if an example in the dataset is y=0 then the domain classifier
    should output the probability of being in each domain such that it matches
    the probability of being in that domain out of the examples that have that
    label 0 in the source domain training data (i.e. P(d|y)).

    Based on:
    https://github.com/akuzeee/AFLAC/blob/master/learner.py
    https://github.com/akuzeee/AFLAC/blob/master/DAN.py
    https://github.com/akuzeee/AFLAC/blob/master/AFLAC.py
    """
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.loss_names = ["fe_tc", "domain", "task", "kl"]
        self.mle_for_p_d_given_y()
        # Not fed to the model, but used in the loss
        self.grl_schedule = models.DannGrlSchedule(self.total_steps)

    def mle_for_p_d_given_y(self):
        """ Compute P(d|y)
        https://github.com/akuzeee/AFLAC/blob/master/AFLAC.py#L14

        Note: doing this rather than mle_for_p_d() since default is "dependent_y"
        in their code https://github.com/akuzeee/AFLAC/blob/master/run.py#L138
        """
        # Get lists of all labels and domains so we can compute how many there
        # are of each
        ys = []
        ds = []

        # The domain is 0 for source 0, 1 for source 1, etc.
        # Note: we use the "eval" train dataset since it doesn't repeat infinitely
        for d, dataset in enumerate(self.source_train_eval_datasets):
            for _, y, _ in dataset:
                ys.append(y)
                ds.append(tf.ones_like(y)*d)

        # Fix Tensorflow bug / problem: expand, transpose, concat, then squeeze.
        # What I wanted to do is just tf.concat(ys, axis=0)... since ys is an
        # array of 1D tensors. But, it gives an error:
        # "Expected concatenating dimensions in the range [0, 0)"
        ys = [tf.transpose(tf.expand_dims(x, axis=0)) for x in ys]
        ds = [tf.transpose(tf.expand_dims(x, axis=0)) for x in ds]
        y = tf.cast(tf.squeeze(tf.concat(ys, axis=0)), dtype=tf.int32)
        d = tf.cast(tf.squeeze(tf.concat(ds, axis=0)), dtype=tf.int32)

        # Convert to numpy to ease converting the AFLAC code
        y = y.numpy()
        d = d.numpy()

        num_y_keys = len(np.unique(y))
        num_d_keys = len(np.unique(d))
        # Note: do <= not == since sometimes a person doesn't perform any of
        # a certain class, so it may be less. Though, for domains it really
        # should be equal unless one of the domains has no data.
        assert num_y_keys <= self.num_classes
        assert num_d_keys <= self.num_source_domains

        # Note: using domain_outputs not num_source_domains, since we have an
        # extra domain label if there's only one source domain.
        p_d_given_y = np.zeros((self.num_classes, self.domain_outputs),
            dtype=np.float32)

        # Classes are numbered 0, 1, ..., num_classes-1
        for y_key in range(self.num_classes):
            indices = np.where(y == y_key)
            d_given_key = d[indices]
            d_keys, d_counts = np.unique(d_given_key, return_counts=True)
            p_d_given_key = np.zeros((self.num_source_domains,))
            p_d_given_key[d_keys] = d_counts
            p_d_given_y[y_key] = p_d_given_key

        # Normalize so for each class, the domain counts sum to one
        p_d_given_y = tf.constant(p_d_given_y, dtype=tf.float32)
        p_d_given_y /= tf.reduce_sum(tf.math.abs(p_d_given_y), axis=1, keepdims=True)

        self.p_d_given_y = p_d_given_y

    def create_model(self, model_name):
        assert len(self.shared_modalities) == 1, \
            "AFLAC only supports one shared modality between domains"

        return models.BasicModel(self.num_classes, self.domain_outputs,
            model_name=model_name)

    def compute_losses(self, xs, task_y_true, domain_y_true, task_y_pred,
            domain_y_pred, fe_output, contrastive_output, auxiliary_data,
            which_model, training):
        task_loss = self.task_loss(task_y_true, task_y_pred)
        d_loss = sum([
            self.domain_loss(domain_y_true, d)
            for d in domain_y_pred
        ])/len(domain_y_pred)

        # Gather the P(d|y) for the true y's for each example.
        # Note: this doesn't leak target-domain label information since this
        # is DG not MS-DA, so we have no data (x or y) for the target domain.
        d_true = tf.gather(self.p_d_given_y, tf.cast(task_y_true, dtype=tf.int32))

        # p_d_given_y (above, now d_true) is already normalized, but
        # domain_y_pred is just "logits" (no softmax in model), so pass the
        # domain_y_pred through softmax before computing KLD.
        #
        # Also, we could implement KL divergence as done in
        # https://github.com/akuzeee/AFLAC/blob/master/utils.py#L183 with
        # something like:
        #   cce = tf.keras.losses.CategoricalCrossentropy()
        #   kl_d = -cce(q, q) + cce(q, p)
        # However, it's equivalent to using the KLD function, so we'll just use
        # that.
        #
        # Pf: -cce(q,q) + cce(q,p)
        #   = sum_i^D q_i log q_i - sum_i^D q_i log p_i (for D domains)
        #   = sum_i^D q_i * (log q_i - log p_i)
        #   = sum_i^D q_i log(q_i/p_i)
        #   = D_KL(q||p)
        # (then of course, this is done for each example in the batch)
        #
        # See:
        # https://en.wikipedia.org/wiki/Cross_entropy
        # https://en.wikipedia.org/wiki/Kullback%E2%80%93Leibler_divergence
        kl_loss = tf.keras.losses.KLD(d_true, tf.nn.softmax(domain_y_pred))

        # Looking at Figure 2 -- https://arxiv.org/pdf/1904.12543.pdf
        # They backpropagate the task and KL (weighted by alpha) losses to FE
        # (and task... but KL doesn't matter for updating the task classifier).
        # They backpropagate the domain loss for updating DC.
        #
        # Their code:
        # https://github.com/akuzeee/AFLAC/blob/master/AFLAC.py#L158
        # Note that y_optimizer only updates FE and TC and d_optimizer only
        # updates DC. Rather than putting in GradMultiplyLayerF into network,
        # I'll just calculate alpha here and weight the KL loss by it since
        # we're ignoring the gradient throughout DC anyway, don't need it to be
        # weighted only through part of the network.
        alpha = self.grl_schedule(self.global_step)
        fe_tc_loss = task_loss + alpha*kl_loss

        return [fe_tc_loss, d_loss, task_loss, kl_loss]

    def compute_gradients(self, tape, losses, which_model):
        fe_tc_loss, d_loss, _, _ = losses
        grad = tape.gradient(fe_tc_loss,
            self.model[which_model].trainable_variables_task_fe)
        d_grad = tape.gradient(d_loss,
            self.model[which_model].trainable_variables_domain)
        return [grad, d_grad]

    def apply_gradients(self, gradients, which_model):
        grad, d_grad = gradients
        self.opt[which_model]["opt"].apply_gradients(zip(grad,
            self.model[which_model].trainable_variables_task_fe))
        self.opt[which_model]["d_opt"].apply_gradients(zip(d_grad,
            self.model[which_model].trainable_variables_domain))


#
# Domain generalization CAL-DG versions, changes:
# - calda -> caldg in name
# - pass domain_generalization=True (so we don't throw out d=0 data)
# - inherit from MethodDannDG rather than MethodDann
#
@register_method("caldg_xs_r")
class MethodCaldgCrossR(MethodCaldaBase, MethodDannDG):
    def __init__(self, *args, **kwargs):
        super().__init__(*args,
            domain_generalization=True, **kwargs)


@register_method("caldg_in_r")
class MethodCaldgInR(MethodCaldaBase, MethodDannDG):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, in_domain=True,
            domain_generalization=True, **kwargs)


@register_method("caldg_any_r")
class MethodCaldgAnyR(MethodCaldaBase, MethodDannDG):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, any_domain=True,
            domain_generalization=True, **kwargs)


@register_method("caldg_xs_h")
class MethodCaldgCrossH(MethodCaldaBase, MethodDannDG):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, hard=True,
            domain_generalization=True, **kwargs)


@register_method("caldg_in_h")
class MethodCaldgInH(MethodCaldaBase, MethodDannDG):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, hard=True, in_domain=True,
            domain_generalization=True, **kwargs)


@register_method("caldg_any_h")
class MethodCaldgAnyH(MethodCaldaBase, MethodDannDG):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, hard=True, any_domain=True,
            domain_generalization=True, **kwargs)


def make_loss(from_logits=True, reduction=None):
    if reduction is None:
        cce = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=from_logits)
    else:
        cce = tf.keras.losses.SparseCategoricalCrossentropy(
            from_logits=from_logits, reduction=reduction)

    def loss(y_true, y_pred):
        return cce(y_true, y_pred)

    return loss
