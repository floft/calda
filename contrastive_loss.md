# Contrastive Loss

Some pseudo code and Python example code used when coding the contrastive loss. Note this may not be completely up-to-date.

## Pseudo Code

### Non-vectorized algorithm

See [Algorithm 1](https://openaccess.thecvf.com/content_CVPR_2020/papers/He_Momentum_Contrast_for_Unsupervised_Visual_Representation_Learning_CVPR_2020_paper.pdf), which uses CrossEntropyLoss rather than manually computing the log/exp. We likewise do that here, though due to needing to perform masking, in the code we actually compute it manually.

    similarity_loss = 0
    anchors = { get non-target z's }
    for each anchor,
        d = domain of anchor
        l = label of anchor
        stop gradient for positives/negatives?
        positive set = { get non-d, l z's }  # with max number of these
        negative set = { get non-d, non-l z's }  # with max number of these

        compute cosine similarity between anchor and each pos/neg

        anchor_similarity_loss = 0
        for each positive,
            logits = cat([[pos], neg], axis=1)
            labels = zeros(len(neg)+1) # positives are 0th
            anchor_similarity_loss += CrossEntropyLoss(
                y_true=labels, y_pred=logits/tau)

        similarity_loss += anchor_similarity_loss / num_positives

    similarity_loss = similarity_loss / num_anchors


### The vectorized algorithm

Vectorize this in a way that makes it capable of being compiled and much
faster. Without compiling the non-vectorized version was ~25 s/iter, i.e.
25*30000/3600/24 = ~8.5 days (and we run thousands of these experiments)
and only has ~5% GPU utilization.

Now it's more like ~0.1 s/iter with more like >80% GPU utilization.

Note: we apply limits with max_{anchors,positives,negatives} since
otherwise this is probably O(num_anchors * num_positives * num_negatives)
which is very large if they're all almost the batch size (just
excluding the target), i.e. O(batch_size^3). V1 didn't have any notion
of anchors, so had much fewer components in the loss.

Simplified version of what's in `MethodCaldaBase._similarity_loss()` in *methods.py*:

    similarity_loss = 0
    anchors = non-target z's
    num_anchors = length of anchors, i.e. shape[0]

    positive_set = mask: for each anchor, which example in the batch is in the positive set
    num_positives = reduce_sum(positive_set) since binary, also depends on num_anchors
    negative_set = mask: for each anchor, which example in the batch is in the negative set
    (skip stopping gradient on those, unless we use MoCo)

    compute cosine similarity, include pos/neg mask to remove unwanted terms in sum

    # Note: we actually manually implement the CE loss to handle the mask
    logits = cat([[pos], neg], axis=1)
    labels = zeros((num_anchors, 1))
    similarity_loss += CE_Loss(labels, logits/tau) / num_positives

## Python Example Code

Examples for `MethodCaldaBase._similarity_loss()` in *methods.py*

### Example computing the positive/negative masks

    >>> import tensorflow as tf
    >>> domain_y_true = tf.constant([0,0,0,1,1,1,2,2,2])
    >>> task_y_true = tf.constant([0,1,2,0,1,2,0,1,2])
    >>> z_output = tf.constant([[1,2],[1,3],[1,4],[2,5],[2,6],[2,7],[3,8],[3,9],[3,10]], dtype=tf.float32)
    >>> nontarget = tf.where(tf.not_equal(domain_y_true, 0))
    >>> y = tf.gather(task_y_true, nontarget, axis=0)
    >>> d = tf.gather(domain_y_true, nontarget, axis=0)
    >>> z = tf.gather(z_output, nontarget, axis=0)
    >>> y = tf.squeeze(y, axis=1)
    >>> d = tf.squeeze(d, axis=1)
    >>> z = tf.squeeze(z, axis=1)
    >>> num_total = tf.shape(y)[0]
    >>> anchors = tf.range(0, num_total)
    >>> num_anchors = tf.shape(anchors)[0]
    >>> def _cartesian_product(a, b):
        tile_a = tf.tile(tf.expand_dims(a, 1), [1, tf.shape(b)[0]])
        tile_a = tf.expand_dims(tile_a, 2)
        tile_b = tf.tile(tf.expand_dims(b, 0), [tf.shape(a)[0], 1])
        tile_b = tf.expand_dims(tile_b, 2)
        cartesian_product = tf.concat([tile_a, tile_b], axis=2)
        return cartesian_product
    >>> prod_indices = _cartesian_product(anchors, tf.range(0, num_total))
    >>> anchor_d = tf.gather_nd(d, tf.expand_dims(prod_indices[:,:,0],axis=-1))
    >>> all_d = tf.gather_nd(d, tf.expand_dims(prod_indices[:,:,1],axis=-1))
    >>> anchor_y = tf.gather_nd(y, tf.expand_dims(prod_indices[:,:,0],axis=-1))
    >>> all_y = tf.gather_nd(y, tf.expand_dims(prod_indices[:,:,1],axis=-1))
    >>> tf.cast(anchor_d!=all_d, tf.int32)
    <tf.Tensor: shape=(6, 6), dtype=int32, numpy=
    array([[0, 0, 0, 1, 1, 1],
            [0, 0, 0, 1, 1, 1],
            [0, 0, 0, 1, 1, 1],
            [1, 1, 1, 0, 0, 0],
            [1, 1, 1, 0, 0, 0],
            [1, 1, 1, 0, 0, 0]], dtype=int32)>
    >>> positives = tf.logical_and(tf.not_equal(anchor_d, all_d), tf.equal(anchor_y, all_y))  # diff domain, same label
    >>> tf.cast(positives, tf.int32)
    <tf.Tensor: shape=(6, 6), dtype=int32, numpy=
    array([[0, 0, 0, 1, 0, 0],
            [0, 0, 0, 0, 1, 0],
            [0, 0, 0, 0, 0, 1],
            [1, 0, 0, 0, 0, 0],
            [0, 1, 0, 0, 0, 0],
            [0, 0, 1, 0, 0, 0]], dtype=int32)>
    >>> negatives = tf.logical_and(tf.not_equal(anchor_d, all_d), tf.not_equal(anchor_y, all_y))  # diff domain, diff label
    >>> tf.cast(negatives, tf.int32)
    <tf.Tensor: shape=(6, 6), dtype=int32, numpy=
    array([[0, 0, 0, 0, 1, 1],
            [0, 0, 0, 1, 0, 1],
            [0, 0, 0, 1, 1, 0],
            [0, 1, 1, 0, 0, 0],
            [1, 0, 1, 0, 0, 0],
            [1, 1, 0, 0, 0, 0]], dtype=int32)>

### Example with in-domain contrastive-learning

    >>> domain_y_true = tf.constant([0,0,0,0,1,1,1,1,2,2,2,2])
    >>> task_y_true = tf.constant([0,1,2,2,0,1,2,1,0,1,2,0])
    >>> ...
    >>> anchors = tf.random.shuffle(anchors)[:5]
    >>> num_anchors = tf.shape(anchors)[0]
    >>> prod_indices = _cartesian_product(anchors, tf.range(0, num_total))
    >>> anchor_d = tf.gather_nd(d, tf.expand_dims(prod_indices[:,:,0],axis=-1))
    >>> all_d = tf.gather_nd(d, tf.expand_dims(prod_indices[:,:,1],axis=-1))
    >>> anchor_y = tf.gather_nd(y, tf.expand_dims(prod_indices[:,:,0],axis=-1))
    >>> all_y = tf.gather_nd(y, tf.expand_dims(prod_indices[:,:,1],axis=-1))
    >>> is_not_anchor = tf.logical_not(tf.cast(tf.gather_nd(tf.eye(num_total), tf.expand_dims(anchors, axis=-1)), dtype=tf.bool))
    >>> positives_indomain = tf.logical_and(tf.logical_and(tf.equal(anchor_d, all_d), tf.equal(anchor_y, all_y)), is_not_anchor)  # same domain, same label, not anchor
    >>> negatives_indomain = tf.logical_and(tf.logical_and(tf.equal(anchor_d, all_d), tf.not_equal(anchor_y, all_y)), is_not_anchor)  # same domain, diff label, not anchor
    >>> anchors
    <tf.Tensor: shape=(5,), dtype=int32, numpy=array([6, 0, 8, 9, 1], dtype=int32)>
    >>> tf.cast(positives_indomain, tf.int32)
    array([[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]], dtype=int32)>
    >>> tf.cast(negatives_indomain, tf.int32)
    <tf.Tensor: shape=(5, 12), dtype=int32, numpy=
    array([[0, 0, 0, 0, 1, 1, 0, 1, 0, 0, 0, 0],
            [0, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 1, 1],
            [1, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0]], dtype=int32)>

### Example computing the cosine similarity matrix

    >>> positives_similarity_indices = tf.gather_nd(prod_indices, tf.where(tf.not_equal(tf.cast(positives, tf.int32), 0)))
    >>> negatives_similarity_indices = tf.gather_nd(prod_indices, tf.where(tf.not_equal(tf.cast(negatives, tf.int32), 0)))
    >>> anchors_indices = positives_similarity_indices[:, 0]
    >>> def cosine_similarity_from_indices(vectors, indices):
        vectors = tf.gather_nd(vectors, tf.expand_dims(indices, axis=-1))
        vectors = tf.math.l2_normalize(vectors, axis=-1)
        return tf.reduce_sum(tf.reduce_prod(vectors, axis=1), axis=-1)
    >>> positives_similarity = cosine_similarity_from_indices(z, positives_similarity_indices)
    >>> negatives_similarity = cosine_similarity_from_indices(z, negatives_similarity_indices)

### Example of computing the loss (and checking it's right)

    >>> num_total = tf.shape(y)[0]
    >>> negatives_square = tf.scatter_nd(negatives_similarity_indices, negatives_similarity, [num_total, num_total])
    >>> negatives_square_mask = tf.scatter_nd(negatives_similarity_indices, tf.ones_like(negatives_similarity), [num_total, num_total])
    >>> negatives_for_anchors = tf.gather(negatives_square, anchors_indices)
    >>> negatives_for_anchors_mask = tf.gather(negatives_square_mask, anchors_indices)
    >>> tau=1
    >>> ce_positive = tf.math.exp(positives_similarity/tau)
    >>> ce_negatives = tf.multiply(tf.math.exp(negatives_for_anchors/tau), negatives_for_anchors_mask)
    >>> total = ce_positive + tf.reduce_sum(ce_negatives, axis=-1)
    >>> cross_entropy_loss_from_logits = - tf.math.log(ce_positive / total)
    >>> cross_entropy_loss_from_logits
    <tf.Tensor: shape=(6,), dtype=float32, numpy=
    array([1.0968751, 1.098231 , 1.0972776, 1.0974636, 1.0977226, 1.0971978],
            dtype=float32)>
    >>> similarity_loss = tf.reduce_sum(cross_entropy_loss_from_logits)

    # check the first value is right
    >>> pos = tf.expand_dims(positives_similarity[0], axis=-1)
    >>> neg = tf.squeeze(tf.gather(negatives_similarity, tf.where(negatives_similarity_indices[:, 0] == 0)), axis=-1)
    >>> pred = tf.expand_dims(tf.concat([pos, neg], axis=0), axis=0)
    >>> loss = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
    >>> true = tf.zeros((1,1), dtype=tf.float32)
    >>> loss(true, pred)
    <tf.Tensor: shape=(), dtype=float32, numpy=1.0968751>
    >>> loss(true, pred) == cross_entropy_loss_from_logits[0]
    <tf.Tensor: shape=(), dtype=bool, numpy=True>


### Example illustrating that the extra zeros in the negatives needs to be accounted for

So, we use tf.vectorized_map (possibly out of date)

    >>> logits = tf.concat([tf.expand_dims(nonzero_positives_similarity, axis=1), negatives_similarity_with_dups], axis=1)
    >>> num_positives = tf.reduce_sum(tf.cast(positives, tf.float32))
    >>> labels = tf.zeros((tf.cast(num_positives, tf.int32), 1))
    >>> loss = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
    >>> true = tf.expand_dims(labels[0], axis=0)
    >>> pred = tf.expand_dims(logits[0], axis=0)
    >>> loss(true, pred)
    <tf.Tensor: shape=(), dtype=float32, numpy=1.9180634>
    >>> pred_no_zeros = tf.gather_nd(pred, tf.where(pred != 0))
    >>> loss(true, pred_no_zeros)
    <tf.Tensor: shape=(), dtype=float32, numpy=1.6018089>


### Example for hard positives/negatives

i.e. sort the positives/anchors lists together based on the highest task loss for the positive at the top and sort the negatives list based on the highest task loss for that
negative at the top

    >>> hardness = tf.constant([8.0,7.0,6.0,5.0,4.0,3.0,2.0,1.0,0.0])
    >>> pos_hardness = tf.gather(hardness, positives_similarity_indices[:, 1])
    >>> neg_hardness = tf.gather(hardness, negatives_similarity_indices[:, 1])
    >>> pos_hardsort = tf.argsort(pos_hardness, axis=-1, direction='DESCENDING', stable=False)
    >>> neg_hardsort = tf.argsort(neg_hardness, axis=-1, direction='DESCENDING', stable=False)
    >>> anchors_indices = positives_similarity_indices[:, 0]
    >>> tf.gather(positives_similarity_indices, pos_hardsort) # positives_similarity_indices
    >>> tf.gather(anchors_indices, pos_hardsort) # anchors_indices
    >>> tf.gather(negatives_similarity_indices, neg_hardsort) # negatives_similarity_indices