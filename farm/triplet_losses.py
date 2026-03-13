import tensorflow as tf


def make_triplet_loss(latent_dim, margin):
    def loss(_, y_pred):
        anchor = y_pred[:, :latent_dim]
        positive = y_pred[:, latent_dim : 2 * latent_dim]
        negative = y_pred[:, 2 * latent_dim :]

        pos_dist = tf.reduce_sum(tf.square(anchor - positive), axis=1)
        neg_dist = tf.reduce_sum(tf.square(anchor - negative), axis=1)
        basic_loss = pos_dist - neg_dist + margin

        return tf.reduce_mean(tf.maximum(basic_loss, 0.0))

    return loss
