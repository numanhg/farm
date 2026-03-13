import os

import numpy as np
from sklearn.preprocessing import LabelEncoder

from farm.preprocessing import preprocess_data
from farm.triplet import TripletAutoencoder


def load_training_subset(npz_path, apply_preprocessing=False, variance_threshold=1e-5):
    data_npz = np.load(npz_path, allow_pickle=True)
    features = data_npz["features"]
    family_names = data_npz["family_names"]
    categories = data_npz["categories"]

    if apply_preprocessing:
        features, _, _ = preprocess_data(features, variance_threshold=variance_threshold)

    malicious_mask = categories == "malicious"

    le = LabelEncoder()
    le.fit(family_names)

    malicious_features = features[malicious_mask]
    malicious_labels = le.transform(family_names[malicious_mask])

    return malicious_features, malicious_labels


def train_triplet_model(
    npz_path,
    weights_path,
    latent_dim=32,
    triplet_epochs=100,
    triplet_learning_rate=1e-4,
    batch_size=32,
    min_samples=None,
    apply_preprocessing=False,
    variance_threshold=1e-5,
    force_train=False,
):
    X_train, y_train = load_training_subset(
        npz_path=npz_path,
        apply_preprocessing=apply_preprocessing,
        variance_threshold=variance_threshold,
    )

    model = TripletAutoencoder(
        X_train=X_train,
        y_train=y_train,
        triplet_weights_path=os.path.abspath(weights_path),
        save_meta_dict=False,
        triplet_epochs=triplet_epochs,
        latent_dim=latent_dim,
        batch_size=batch_size,
        triplet_learning_rate=triplet_learning_rate,
        dbscan_min_samples=min_samples,
        force_train=force_train,
    )
    return model
