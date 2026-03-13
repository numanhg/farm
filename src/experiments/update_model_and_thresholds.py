import argparse
import os
from collections import Counter

import matplotlib.pyplot as plt
import numpy as np
from kneed import KneeLocator
from scipy.stats import probplot
from sklearn.cluster import DBSCAN
from sklearn.model_selection import train_test_split
from sklearn.neighbors import NearestNeighbors

from meine import TripletAutoencoder


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--prototype_label_id", type=int, default=0)
    return parser.parse_args()


def load_retrain_data(prototype_label_id: int):
    retrain_data = np.load(f"retrain_{prototype_label_id}.npz")
    return retrain_data["X_retrain"], retrain_data["y_retrain"]


def sample_training_data(
    features: np.ndarray,
    labels: np.ndarray,
    prototype_label_id: int,
    per_class_size: int = 50,
):
    unique_classes = np.unique(labels)
    sampled_features = []
    sampled_labels = []

    for cls in unique_classes:
        cls_indices = np.where(labels == cls)[0]
        if cls == prototype_label_id or len(cls_indices) <= per_class_size:
            selected_indices = cls_indices
        else:
            selected_indices = np.random.choice(cls_indices, size=per_class_size, replace=False)

        sampled_features.append(features[selected_indices])
        sampled_labels.append(labels[selected_indices])

    return np.concatenate(sampled_features, axis=0), np.concatenate(sampled_labels, axis=0)


def train_updated_model(malicious_features: np.ndarray, malicious_labels: np.ndarray, prototype_label_id: int):
    return TripletAutoencoder(
        X_train=malicious_features,
        y_train=malicious_labels,
        triplet_initial_weights=os.path.abspath("model.weights.h5"),
        triplet_weights_path=os.path.abspath(f"model_updated_{prototype_label_id}.weights.h5"),
        triplet_epochs=30,
        triplet_learning_rate=1e-4,
        force_train=True,
        freeze_encoder=True,
    )


def split_train_val_test(features: np.ndarray, labels: np.ndarray):
    x_temp, x_test, y_temp, y_test = train_test_split(
        features,
        labels,
        test_size=0.2,
        random_state=42,
    )
    x_train, x_val, y_train, y_val = train_test_split(
        x_temp,
        y_temp,
        test_size=0.25,
        random_state=42,
    )
    return np.vstack((x_train, x_val)), np.hstack((y_train, y_val)), x_test, y_test


def find_eps_with_knee(data: np.ndarray, min_samples: int = 5, plot: bool = False, class_label=None):
    neighbors = NearestNeighbors(n_neighbors=min_samples)
    neighbors_fit = neighbors.fit(data)
    distances, _ = neighbors_fit.kneighbors(data)
    k_distances = np.sort(distances[:, -1])

    knee = KneeLocator(range(len(k_distances)), k_distances, curve="convex", direction="increasing")
    eps_candidate = k_distances[knee.knee] if knee.knee else None

    if plot:
        plt.figure(figsize=(8, 4))
        plt.plot(k_distances, label="k-distance")
        if knee.knee:
            plt.axvline(knee.knee, color="r", linestyle="--", label=f"Knee at eps ≈ {eps_candidate:.3f}")
        plt.title(f"Elbow Method for Class {class_label}")
        plt.xlabel("Points sorted by distance")
        plt.ylabel(f"{min_samples}-NN Distance")
        plt.legend()
        plt.grid(True)
        plt.show()

    return eps_candidate


def compute_eps_per_class(x_train_features: np.ndarray, y_train: np.ndarray, min_samples: int):
    eps_dict = {}

    for class_label in np.unique(y_train):
        x_single = x_train_features[y_train == class_label]

        if len(x_single) < min_samples:
            eps_dict[class_label] = 0.5
            continue

        eps = find_eps_with_knee(x_single, min_samples=min_samples, class_label=class_label, plot=False)
        if eps is None or eps <= 0:
            eps = 0.5

        dbscan = DBSCAN(eps=eps, min_samples=min_samples).fit(x_single)
        print(f"Class: {class_label}: {Counter(dbscan.labels_)}")
        eps_dict[class_label] = float(eps)

    return eps_dict


def calculate_threshold(
    eps_dict,
    x_train: np.ndarray,
    y_train: np.ndarray,
    min_samples: int = 5,
    debug: bool = False,
):
    thresholds = {}
    centroids_dict = {}

    for lbl in np.unique(y_train):
        single_class_features = x_train[y_train == lbl]
        if len(single_class_features) < min_samples:
            centroid = np.mean(single_class_features, axis=0)
            distances = np.linalg.norm(single_class_features - centroid, axis=1)
            threshold = float(np.max(distances)) if len(distances) else 0.0
            thresholds[lbl] = threshold
            centroids_dict[lbl] = np.array([centroid])
            continue

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

            if debug:
                probplot(distances, dist="norm", plot=plt)
                plt.title(f"Q-Q Plot | Class {lbl}, Cluster {cluster}")
                plt.grid(True)
                plt.show()

            threshold = np.max(distances)

            class_thresholds.append(float(threshold))

        thresholds[lbl] = max(class_thresholds)
        centroids_dict[lbl] = np.array(centroids)

    return thresholds, centroids_dict


def save_artifacts(prototype_label_id: int, centroids_dict, thresholds_dict, eps_dict) -> None:
    centroids_out = {str(k): v for k, v in centroids_dict.items()}
    thresholds_out = {str(k): v for k, v in thresholds_dict.items()}
    eps_out = {str(k): v for k, v in eps_dict.items()}

    np.savez(f"centroids_{prototype_label_id}.npz", **centroids_out)
    np.savez(f"thresholds_{prototype_label_id}.npz", **thresholds_out)
    np.savez(f"epsilons_{prototype_label_id}.npz", **eps_out)


def main() -> None:
    args = parse_args()
    prototype_label_id = args.prototype_label_id

    malicious_features, malicious_labels = load_retrain_data(prototype_label_id)
    malicious_features, malicious_labels = sample_training_data(
        malicious_features,
        malicious_labels,
        prototype_label_id,
    )

    print(f"Malicious features shape: {malicious_features.shape}")
    print(f"Malicious labels shape: {malicious_labels.shape}")

    triplet_model = train_updated_model(malicious_features, malicious_labels, prototype_label_id)

    x_train, y_train, _, _ = split_train_val_test(malicious_features, malicious_labels)
    x_train_features = triplet_model.encoder_model.predict(x_train)

    min_samples = 31
    eps_dict = compute_eps_per_class(x_train_features, y_train, min_samples=min_samples)

    thresholds_dict, centroids_dict = calculate_threshold(
        eps_dict=eps_dict,
        x_train=x_train_features,
        y_train=y_train,
        min_samples=min_samples,
        debug=False,
    )

    save_artifacts(prototype_label_id, centroids_dict, thresholds_dict, eps_dict)


if __name__ == "__main__":
    main()
