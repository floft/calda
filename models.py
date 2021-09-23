"""
Models
"""
import tensorflow as tf

from absl import flags

FLAGS = flags.FLAGS

models = {}


def register_model(name):
    """ Add model to the list of models, e.g. add @register_model("name")
    before a class definition """
    assert name not in models, "duplicate model named " + name

    def decorator(cls):
        models[name] = cls
        return cls

    return decorator


def get_model(name, *args, **kwargs):
    """ Based on the given name, call the correct model """
    assert name in models.keys(), \
        "Unknown model name " + name
    return models[name](*args, **kwargs)


def list_models():
    """ Returns list of all the available models """
    return list(models.keys())


@tf.custom_gradient
def flip_gradient(x, grl_lambda):
    """ Forward pass identity, backward pass negate gradient and multiply by  """
    grl_lambda = tf.cast(grl_lambda, dtype=tf.float32)

    def grad(dy):
        # the 0 is for grl_lambda, which doesn't have a gradient
        return tf.negative(dy) * grl_lambda * tf.ones_like(x), 0

    return x, grad


class FlipGradient(tf.keras.layers.Layer):
    """
    Gradient reversal layer

    global_step = tf.Variable storing the current step
    schedule = a function taking the global_step and computing the grl_lambda,
        e.g. `lambda step: 1.0` or some more complex function.
    """
    def __init__(self, global_step, grl_schedule, **kwargs):
        super().__init__(**kwargs)
        self.global_step = global_step
        self.grl_schedule = grl_schedule

    def call(self, inputs, **kwargs):
        """ Calculate grl_lambda first based on the current global step (a
        variable) and then create the layer that does nothing except flip
        the gradients """
        grl_lambda = self.grl_schedule(self.global_step)
        return flip_gradient(inputs, grl_lambda)


def DannGrlSchedule(num_steps):
    """ GRL schedule from DANN paper """
    num_steps = tf.cast(num_steps, tf.float32)

    def schedule(step):
        step = tf.cast(step, tf.float32)
        return 2/(1+tf.exp(-10*(step/(num_steps+1))))-1

    return schedule


class StopGradient(tf.keras.layers.Layer):
    """ Stop gradient layer """
    def call(self, inputs, **kwargs):
        return tf.stop_gradient(inputs)


class ModelBase(tf.keras.Model):
    """ Base model class (inheriting from Keras' Model class) """
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def _get_trainable_variables_list(self, model_list):
        """ Get all trainable variables if model is a list """
        model_vars = []

        for m in model_list:
            model_vars += m.trainable_variables

        return model_vars

    def _get_trainable_variables(self, model):
        """ Get trainable variables if model is a list or not """
        if isinstance(model, list):
            return self._get_trainable_variables_list(model)

        return model.trainable_variables

    @property
    def trainable_variables_fe(self):
        return self._get_trainable_variables(self.feature_extractor)

    @property
    def trainable_variables_task(self):
        return self._get_trainable_variables(self.task_classifier)

    @property
    def trainable_variables_domain(self):
        return self._get_trainable_variables(self.domain_classifier)

    @property
    def trainable_variables_contrastive(self):
        return self._get_trainable_variables(self.contrastive_head)

    @property
    def trainable_variables_task_fe(self):
        variables = self.trainable_variables_fe \
            + self.trainable_variables_task

        if self.contrastive_head is not None:
            variables += self.trainable_variables_contrastive

        return variables

    @property
    def trainable_variables_task_fe_domain(self):
        variables = self.trainable_variables_fe \
            + self.trainable_variables_task \
            + self.trainable_variables_domain

        if self.contrastive_head is not None:
            variables += self.trainable_variables_contrastive

        return variables

    @property
    def trainable_variables(self):
        """ Returns all trainable variables in the model """
        return self.trainable_variables_task_fe_domain

    # Allow easily overriding each part of the call() function, without having
    # to override call() in its entirety
    def call_feature_extractor(self, inputs, which_fe=None, which_tc=None,
            which_dc=None, **kwargs):
        if which_fe is not None:
            assert isinstance(self.feature_extractor, list)
            return self.feature_extractor[which_fe](inputs, **kwargs)

        assert not isinstance(self.feature_extractor, list)
        return self.feature_extractor(inputs, **kwargs)

    def call_task_classifier(self, fe, which_fe=None, which_tc=None,
            which_dc=None, **kwargs):
        if which_tc is not None:
            assert isinstance(self.task_classifier, list)
            return self.task_classifier[which_tc](fe, **kwargs)

        assert not isinstance(self.task_classifier, list)
        return self.task_classifier(fe, **kwargs)

    def call_domain_classifier(self, fe, task, which_fe=None, which_tc=None,
            which_dc=None, **kwargs):
        if which_dc is not None:
            assert isinstance(self.domain_classifier, list)
            return self.domain_classifier[which_dc](fe, **kwargs)

        assert not isinstance(self.domain_classifier, list)
        return self.domain_classifier(fe, **kwargs)

    def call_contrastive_head(self, fe, which_fe=None, which_tc=None,
            which_dc=None, which_ch=None, **kwargs):
        if which_ch is not None:
            assert isinstance(self.contrastive_head, list)
            return self.contrastive_head[which_ch](fe, **kwargs)

        assert not isinstance(self.contrastive_head, list)
        return self.contrastive_head(fe, **kwargs)

    def call(self, inputs, training=None, input_is_list=True, **kwargs):
        # For backwards compatibility, the FE and DC aren't always lists, e.g.
        # for some methods that don't currently support multiple modalities.
        if isinstance(self.feature_extractor, list) and \
                isinstance(self.domain_classifier, list):
            # We have a separate feature extractor for each modality. We use the
            # lowest index ones first, i.e. if for example the source has 2 and the
            # target has 1, the lowest index source FE will be used for the target
            # modality ("left to right" of sorts). Change the ordering with
            # {source,target}_modality_subset if desired.
            assert len(inputs) <= len(self.feature_extractor), \
                "need one feature extractor per modality"

            fe = [
                self.call_feature_extractor(inputs[i], which_fe=i,
                    training=training, **kwargs)
                for i in range(len(inputs))
            ]

            # Concatenate the shared modality FEs together for the task classifier
            # and contrastive head
            fe_concat = tf.concat([fe[i] for i in self.shared_modalities], axis=-1)
            task = self.call_task_classifier(fe_concat,
                training=training, **kwargs)

            if self.contrastive_head is not None:
                contrastive = self.call_contrastive_head(fe_concat,
                    training=training, **kwargs)
            else:
                contrastive = 0.0

            # Pass each FE to the corresponding domain classifier.
            assert len(self.shared_modalities) == len(self.domain_classifier), \
                "need one domain classifier per shared modality"
            assert len(self.shared_modalities) <= len(fe), \
                "need at least the number of shared modalities as inputs"

            domain = [
                self.call_domain_classifier(fe[i], task, which_dc=i,
                    training=training, **kwargs)
                for i in self.shared_modalities
            ]
        else:
            # This is for ART which errors if the input is a list, i.e. if this is
            # false only single-modality is supported and when passing in x, don't
            # do [x] for single-modality.
            if input_is_list:
                assert len(inputs) == 1, \
                    "if more than one modality, must use multiple FE and DC " \
                    "input shape is " + str(inputs.shape)

                # There's only one modality, so just use that one
                inputs = inputs[0]

            fe = self.call_feature_extractor(inputs, training=training, **kwargs)
            task = self.call_task_classifier(fe, training=training, **kwargs)
            domain = self.call_domain_classifier(fe, task, training=training, **kwargs)

            if self.contrastive_head is not None:
                contrastive = self.call_contrastive_head(fe, training=training, **kwargs)
            else:
                contrastive = 0.0

            # Make consistent with the multi-modality case
            domain = [domain]
            fe = [fe]

        # Note: domain and fe are lists
        return task, domain, fe, contrastive


class ModelMakerBase:
    """
    Make the feature extractor, task classifier, and domain classifier models

    This is a class instead of just a make_xyz_model() returning the 3 parts
    because in some cases (e.g. Heterogeneous DA) where we need multiple FE's
    or (e.g. DannSmoothModel) where we need multiple DC's.

    Also, this allows for sharing similar task/domain classifiers used in
    multiple models.
    """
    def __init__(self, **kwargs):
        pass

    def make_feature_extractor(self, **kwargs):
        raise NotImplementedError("must implement for ModelMaker class")

    def make_task_classifier(self, num_classes, **kwargs):
        raise NotImplementedError("must implement for ModelMaker class")

    def make_domain_classifier(self, num_domains, **kwargs):
        raise NotImplementedError("must implement for ModelMaker class")

    def make_contrastive_head(self, num_units, **kwargs):
        raise NotImplementedError("must implement for ModelMaker class")


class CodatsModelMakerBase(ModelMakerBase):
    """ Task and domain classifiers used for CoDATS and thus used for a number
    of these models """
    def make_task_classifier(self, num_classes, **kwargs):
        return tf.keras.Sequential([
            tf.keras.layers.Dense(num_classes),
        ])

    def make_domain_classifier(self, num_domains, **kwargs):
        return tf.keras.Sequential([
            # Note: alternative is Dense(128, activation="tanh") like used by
            # https://arxiv.org/pdf/1902.09820.pdf They say dropout of 0.7 but
            # I'm not sure if that means 1-0.7 = 0.3 or 0.7 itself.
            tf.keras.layers.Dense(500, use_bias=False),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.Activation("relu"),
            tf.keras.layers.Dropout(0.3),

            tf.keras.layers.Dense(500, use_bias=False),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.Activation("relu"),
            tf.keras.layers.Dropout(0.3),

            tf.keras.layers.Dense(num_domains),
        ])

    def make_contrastive_head(self, num_units, **kwargs):
        return tf.keras.Sequential([
            tf.keras.layers.Dense(num_units),
        ])


@register_model("fcn")
class FcnModelMaker(CodatsModelMakerBase):
    """
    FCN (fully CNN) -- but domain classifier has additional dense layers

    From: https://arxiv.org/pdf/1611.06455.pdf
    Tested in: https://arxiv.org/pdf/1809.04356.pdf
    Code from: https://github.com/hfawaz/dl-4-tsc/blob/master/classifiers/fcn.py
    """
    def make_feature_extractor(self, previous_model=None, **kwargs):
        # Make a new feature extractor if no previous feature extractor
        if previous_model is None:
            return tf.keras.Sequential([
                tf.keras.layers.Conv1D(filters=128, kernel_size=8, padding="same",
                    use_bias=False),
                tf.keras.layers.BatchNormalization(),
                tf.keras.layers.Activation("relu"),

                tf.keras.layers.Conv1D(filters=256, kernel_size=5, padding="same",
                    use_bias=False),
                tf.keras.layers.BatchNormalization(),
                tf.keras.layers.Activation("relu"),

                tf.keras.layers.Conv1D(filters=128, kernel_size=3, padding="same",
                    use_bias=False),
                tf.keras.layers.BatchNormalization(),
                tf.keras.layers.Activation("relu"),

                tf.keras.layers.GlobalAveragePooling1D(),
            ])

        # Only totally separate layer is the first Conv1D layer since the
        # input shape may be different. The rest of the layers will be the
        # layers from the other model.
        return tf.keras.Sequential([
            tf.keras.layers.Conv1D(filters=128, kernel_size=8, padding="same",
                    use_bias=False)
        ] + previous_model.layers[1:])


class CnnModelBase(ModelBase):
    """
    Support a variety of CNN-based models, pick via command-line argument

    Also supports having multiple FE's, TC's, or DC's. If not None, then the
    corresponding variable is a list.
    """
    def __init__(self, num_classes, num_domains, model_name,
            num_feature_extractors=None,
            num_task_classifiers=None,
            num_domain_classifiers=None,
            shared_modalities=None,
            share_most_weights=False,
            num_contrastive_units=None,
            num_contrastive_heads=None,
            **kwargs):
        super().__init__(**kwargs)
        self.num_classes = num_classes
        self.num_domains = num_domains
        self.shared_modalities = shared_modalities

        model_maker = get_model(model_name)

        self.feature_extractor = self._make_single_or_multiple(
            model_maker.make_feature_extractor,
            num_feature_extractors, share_most_weights)
        self.task_classifier = self._make_single_or_multiple(
            lambda **kwargs: model_maker.make_task_classifier(num_classes, **kwargs),
            num_task_classifiers, share_most_weights)
        self.domain_classifier = self._make_single_or_multiple(
            lambda **kwargs: model_maker.make_domain_classifier(num_domains, **kwargs),
            num_domain_classifiers, share_most_weights)

        if num_contrastive_units is not None:
            self.contrastive_head = self._make_single_or_multiple(
                lambda **kwargs: model_maker.make_contrastive_head(num_contrastive_units, **kwargs),
                num_contrastive_heads, share_most_weights)
        else:
            self.contrastive_head = None

    def _make_single_or_multiple(self, f, num, share_most_weights):
        if num is not None:
            if share_most_weights:
                # Share most weights via passing in the previous model
                # Note: only used for in feature extractor creation.
                results = []

                for _ in range(num):
                    previous_model = None

                    if len(results) > 0:
                        previous_model = results[-1]

                    results.append(f(previous_model=previous_model))

                return results
            else:
                return [f() for _ in range(num)]

        return f()


class BasicModel(CnnModelBase):
    """ Model without adaptation (i.e. no DANN) """
    pass


class DannModelBase:
    """ DANN adds a gradient reversal layer before the domain classifier

    Note: we don't inherit from CnnModelBase or any other specific model because
    we want to support either CnnModelBase, RnnModelBase, etc. with multiple
    inheritance.
    """
    def __init__(self, num_classes, num_domains, global_step,
            total_steps, **kwargs):
        super().__init__(num_classes, num_domains, **kwargs)
        grl_schedule = DannGrlSchedule(total_steps)
        self.flip_gradient = FlipGradient(global_step, grl_schedule)

    def call_domain_classifier(self, fe, task, **kwargs):
        # Pass FE output through GRL then to DC
        grl_output = self.flip_gradient(fe, **kwargs)
        return super().call_domain_classifier(grl_output, task, **kwargs)


class DannModel(DannModelBase, CnnModelBase):
    """ Model with adaptation (i.e. with DANN) """
    pass


class SleepModel(DannModelBase, CnnModelBase):
    """ Sleep model is DANN but concatenating task classifier output (with stop
    gradient) with feature extractor output when fed to the domain classifier """
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.concat = tf.keras.layers.Concatenate(axis=1)
        self.stop_gradient = StopGradient()

    def call_domain_classifier(self, fe, task, **kwargs):
        # We could support this but it's awkward since we want to call the super's
        # super's call_domain_classifier but not the super's version...
        assert not isinstance(self.domain_classifier, list), \
            "currently do not support SleepModel with multiple domain classifiers"

        # Pass FE output through GRL and append stop-gradient-ed task output too
        grl_output = self.flip_gradient(fe, **kwargs)
        task_stop_gradient = self.stop_gradient(task)
        domain_input = self.concat([grl_output, task_stop_gradient])
        return self.domain_classifier(domain_input, **kwargs)
