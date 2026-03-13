import numpy as np


def sample_triplet(X, y):
    classes = np.unique(y)
    anchor_class = np.random.choice(classes)
    negative_class = np.random.choice(classes[classes != anchor_class])

    anchor_index = np.random.choice(np.where(y == anchor_class)[0])
    positive_index = np.random.choice(np.where(y == anchor_class)[0])
    negative_index = np.random.choice(np.where(y == negative_class)[0])

    return X[anchor_index], X[positive_index], X[negative_index]


def triplet_generator(X, y, batch_size, latent_dim):
    while True:
        anchors, positives, negatives, mse_targets = [], [], [], []

        for _ in range(batch_size):
            anchor, positive, negative = sample_triplet(X, y)
            anchors.append(anchor)
            positives.append(positive)
            negatives.append(negative)
            mse_targets.append(anchor)

        yield [
            np.array(anchors),
            np.array(positives),
            np.array(negatives),
        ], [
            np.zeros((batch_size, 3 * latent_dim)),
            np.array(mse_targets),
        ]
