import os
from collections import Counter

import numpy as np
from kneed import KneeLocator
from sklearn.cluster import DBSCAN
from sklearn.neighbors import NearestNeighbors


def find_eps_with_knee(data, min_samples):
    if len(data) < 2:
        return 0.0

    n_neighbors = min(min_samples, len(data))
    neighbors = NearestNeighbors(n_neighbors=n_neighbors)
    neighbors_fit = neighbors.fit(data)
    distances, _ = neighbors_fit.kneighbors(data)
    k_distances = np.sort(distances[:, -1])

    knee = KneeLocator(range(len(k_distances)), k_distances, curve="convex", direction="increasing")
    return float(k_distances[knee.knee]) if knee.knee else float(k_distances[-1])


def create_epsilon_dict(embedded_features, labels, min_samples, log_clusters=False):
    eps_dict = {}

    for class_label in np.unique(labels):
        class_features = embedded_features[labels == class_label]
        eps = find_eps_with_knee(class_features, min_samples=min_samples)
        eps_dict[class_label] = eps

        if log_clusters:
            dbscan = DBSCAN(eps=eps, min_samples=min_samples).fit(class_features)
            print(f"Class: {class_label}: {Counter(dbscan.labels_)}")

    return eps_dict


def calculate_thresholds_and_centroids(eps_dict, embedded_features, labels, min_samples):
    thresholds = {}
    centroids_dict = {}

    for lbl in np.unique(labels):
        single_class_features = embedded_features[labels == lbl]
        dbscan = DBSCAN(eps=eps_dict[lbl], min_samples=min_samples).fit(single_class_features)

        cluster_labels = dbscan.labels_
        unique_clusters = set(cluster_labels) - {-1}

        centroids = []
        class_thresholds = []

        for cluster in unique_clusters:
            cluster_points = single_class_features[cluster_labels == cluster]
            cluster_centroid = np.mean(cluster_points, axis=0)
            centroids.append(cluster_centroid)

            distances = np.linalg.norm(cluster_points - cluster_centroid, axis=1)
            class_thresholds.append(np.max(distances))

        thresholds[lbl] = max(class_thresholds) if class_thresholds else 0.0
        centroids_dict[lbl] = np.array(centroids)

    return thresholds, centroids_dict


def build_meta_dicts(encoder_model, X_train, y_train, X_val=None, y_val=None, min_samples=5, log_clusters=False):
    if X_val is not None and y_val is not None:
        X_fit = np.vstack((X_train, X_val))
        y_fit = np.hstack((y_train, y_val))
    else:
        X_fit = X_train
        y_fit = y_train

    embedded_features = encoder_model.predict(X_fit)

    eps_dict = create_epsilon_dict(
        embedded_features=embedded_features,
        labels=y_fit,
        min_samples=min_samples,
        log_clusters=log_clusters,
    )

    thresholds_dict, centroids_dict = calculate_thresholds_and_centroids(
        eps_dict=eps_dict,
        embedded_features=embedded_features,
        labels=y_fit,
        min_samples=min_samples,
    )

    return thresholds_dict, centroids_dict, eps_dict


def save_meta_dicts(thresholds_dict, centroids_dict, eps_dict, save_dir=None):
    if save_dir is None:
        save_dir = os.getcwd()
    os.makedirs(save_dir, exist_ok=True)

    data_to_save = {
        "centroids.npz": centroids_dict,
        "thresholds.npz": thresholds_dict,
        "epsilons.npz": eps_dict,
    }

    for filename, data_dict in data_to_save.items():
        converted_dict = {str(k): v for k, v in data_dict.items()}
        filepath = os.path.join(save_dir, filename)
        np.savez(filepath, **converted_dict)
        print(f"Saved: {filepath}")
