import argparse
import os
from collections import defaultdict

import numpy as np
from scipy.spatial.distance import euclidean
from sklearn.cluster import DBSCAN
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import LabelEncoder

from meine import TripletAutoencoder


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--prototype_label_id", type=int, default=0)
    return parser.parse_args()


def load_data():
    data_npz = np.load("../data/mfc_features_no_packed_timestamp_processed.npz")
    thresholds = np.load("./thresholds.npz", allow_pickle=True)
    centroids = np.load("./centroids.npz", allow_pickle=True)
    epsilons = np.load("./epsilons.npz", allow_pickle=True)

    features = data_npz["features"]
    family_names = data_npz["family_names"]
    categories = data_npz["categories"]

    malicious_mask = categories == "malicious"
    evolving_mask = categories == "malicious-evolving"
    unseen_mask = categories == "malicious-unseen"

    le = LabelEncoder()
    le.fit(family_names)

    return {
        "features": features,
        "family_names": family_names,
        "categories": categories,
        "malicious_features": features[malicious_mask],
        "evolving_features": features[evolving_mask],
        "unseen_features": features[unseen_mask],
        "malicious_labels": le.transform(family_names[malicious_mask]),
        "evolving_labels": le.transform(family_names[evolving_mask]),
        "unseen_labels": le.transform(family_names[unseen_mask]),
        "label_encoder": le,
        "thresholds": thresholds,
        "centroids": centroids,
        "epsilons": epsilons,
    }


def build_model(malicious_features: np.ndarray, malicious_labels: np.ndarray) -> TripletAutoencoder:
    return TripletAutoencoder(
        X_train=malicious_features,
        y_train=malicious_labels,
        triplet_weights_path=os.path.abspath("model.weights.h5"),
        triplet_epochs=100,
    )


def get_eval_set(
    encoder_model,
    x_test: np.ndarray,
    y_test: np.ndarray,
    prototype_features: np.ndarray,
    prototype_labels: np.ndarray,
    original_x_test: np.ndarray,
    label_id: int,
):
    encoded = encoder_model.predict(prototype_features)
    prototype_indices = np.where(prototype_labels == label_id)[0]
    selected_encoded = encoded[prototype_indices]
    selected_original = prototype_features[prototype_indices]
    selected_labels = prototype_labels[prototype_indices]

    x_eval = np.vstack([x_test, selected_encoded])
    x_eval_original = np.vstack([original_x_test, selected_original])
    y_eval = np.hstack([y_test, selected_labels])

    return x_eval, y_eval, prototype_indices, x_eval_original


def evaluate_model(
    features: np.ndarray,
    padded_prototype_indices: np.ndarray,
    centroids,
    thresholds,
):
    predictions, status, drifted_indices = [], [], []
    for idx, feature in enumerate(features):
        min_dist = float("inf")
        assigned_class = None

        for class_label, class_centroids in centroids.items():
            for class_centroid in class_centroids:
                dist = euclidean(class_centroid, feature)
                if dist < min_dist:
                    min_dist = dist
                    assigned_class = class_label

        predictions.append(assigned_class)

        if min_dist > thresholds[str(assigned_class)]:
            status.append("drift")
            drifted_indices.append(padded_prototype_indices[idx])
        else:
            status.append("no-drift")

    return np.array(predictions), np.array(status), np.array(drifted_indices)


def get_cluster_features(features: np.ndarray, epsilons, min_samples: int = 10) -> np.ndarray:
    if len(features) == 0:
        return features

    std_epsilon = np.std(list(epsilons.values())) or 0.5
    current_size = min(min_samples, len(features))

    while True:
        current_features = features[:current_size]

        while len(current_features) <= len(features):
            dbscan = DBSCAN(eps=std_epsilon, min_samples=min_samples).fit(current_features)
            labels = dbscan.labels_

            for label in set(labels):
                if label != -1 and int(np.sum(labels == label)) >= min_samples:
                    return current_features[labels == label]

            if len(current_features) < len(features):
                next_index = len(current_features)
                current_features = np.vstack([current_features, features[next_index]])
            else:
                break

        if current_size == len(features):
            return features

        current_size = min(current_size + 1, len(features))
        std_epsilon += 0.1


def compute_mean_prototypes(embeddings: np.ndarray, labels: np.ndarray):
    class_prototypes = defaultdict(list)
    for cls in np.unique(labels):
        class_embeddings = embeddings[labels == cls]
        if len(class_embeddings) > 0:
            class_prototypes[cls].append(np.mean(class_embeddings, axis=0))
    return class_prototypes


def merge_prototypes(known_prototypes, drift_prototypes):
    all_class_prototypes = defaultdict(list)
    for cls, protos in known_prototypes.items():
        all_class_prototypes[cls].extend(protos)
    for cls, protos in drift_prototypes.items():
        all_class_prototypes[cls].extend(protos)
    return all_class_prototypes


def classify_with_multi_prototypes(embeddings: np.ndarray, class_prototypes_dict) -> np.ndarray:
    all_prototypes = []
    prototype_labels = []

    for cls, protos in class_prototypes_dict.items():
        for proto in protos:
            all_prototypes.append(proto)
            prototype_labels.append(cls)

    all_prototypes = np.array(all_prototypes)
    distances = np.linalg.norm(embeddings[:, np.newaxis] - all_prototypes, axis=2)
    nearest_indices = np.argmin(distances, axis=1)
    return np.array(prototype_labels)[nearest_indices]


def evaluate_adaptation(y_true: np.ndarray, y_pred: np.ndarray, prototype_label_id: int, label_encoder: LabelEncoder):
    evaluation_mask = y_true == prototype_label_id
    prototype_label_accuracy = accuracy_score(y_true[evaluation_mask], y_pred[evaluation_mask])
    overall_accuracy = accuracy_score(y_true, y_pred)

    return {
        "prototype_label": label_encoder.inverse_transform([prototype_label_id])[0],
        "prototype_label_accuracy": prototype_label_accuracy,
        "overall_accuracy": overall_accuracy,
        "num_prototype_samples": int(np.sum(evaluation_mask)),
    }


def collect_correct_prototype_samples(
    x_eval_original: np.ndarray,
    y_true: np.ndarray,
    y_pred: np.ndarray,
    prototype_label_id: int,
    padded_prototype_indices: np.ndarray,
    limit: int = 70,
):
    correct_features = []
    retrain_drifted_indices = []

    for i, (true_label, pred_label) in enumerate(zip(y_true, y_pred)):
        if padded_prototype_indices[i] == -1:
            continue
        if true_label == prototype_label_id and pred_label == prototype_label_id:
            correct_features.append(x_eval_original[i])
            retrain_drifted_indices.append(padded_prototype_indices[i])
            if len(correct_features) == limit:
                break

    if not correct_features:
        return 0, None, None

    return len(correct_features), np.array(correct_features), np.array(retrain_drifted_indices)


def main() -> None:
    args = parse_args()
    prototype_label_id = args.prototype_label_id

    data = load_data()
    triplet_model = build_model(data["malicious_features"], data["malicious_labels"])

    encoder_model = triplet_model.encoder_model
    x_train = encoder_model.predict(triplet_model.X_train)
    y_train = triplet_model.y_train
    x_test = encoder_model.predict(triplet_model.X_test)
    y_test = triplet_model.y_test

    x_eval, y_eval, prototype_indices, x_eval_original = get_eval_set(
        encoder_model=encoder_model,
        x_test=x_test,
        y_test=y_test,
        prototype_features=data["evolving_features"],
        prototype_labels=data["evolving_labels"],
        original_x_test=triplet_model.X_test,
        label_id=prototype_label_id,
    )

    padded_prototype_indices = np.concatenate(
        (np.full(len(x_eval) - len(prototype_indices), -1), prototype_indices)
    )

    _, status, _ = evaluate_model(
        features=x_eval,
        padded_prototype_indices=padded_prototype_indices,
        centroids=data["centroids"],
        thresholds=data["thresholds"],
    )

    drifted_features = x_eval[status == "drift"]
    cluster_features = get_cluster_features(drifted_features, data["epsilons"], min_samples=10)

    known_prototypes = compute_mean_prototypes(x_train, y_train)
    drift_prototypes = compute_mean_prototypes(
        cluster_features,
        np.full(len(cluster_features), prototype_label_id),
    )

    all_class_prototypes = merge_prototypes(known_prototypes, drift_prototypes)
    y_pred = classify_with_multi_prototypes(x_eval, all_class_prototypes)

    for key in all_class_prototypes.keys():
        print(len(all_class_prototypes[key]))

    metrics = evaluate_adaptation(y_eval, y_pred, prototype_label_id, data["label_encoder"])
    print("Before: ", metrics["prototype_label_accuracy"])

    count, correct_samples, retrain_drifted_indices = collect_correct_prototype_samples(
        x_eval_original,
        y_eval,
        y_pred,
        prototype_label_id,
        padded_prototype_indices,
    )

    if correct_samples is None:
        correct_samples = np.empty((0, triplet_model.X_train.shape[1]))
        retrain_drifted_indices = np.empty((0,), dtype=int)

    x_retrain = np.vstack([triplet_model.X_train, correct_samples])
    y_retrain = np.hstack([y_train, np.full(count, prototype_label_id)])

    np.savez_compressed(
        f"retrain_{prototype_label_id}.npz",
        X_retrain=x_retrain,
        y_retrain=y_retrain,
        retrain_drifted_indices=np.array(retrain_drifted_indices),
        padded_protoype_indices=padded_prototype_indices,
    )


if __name__ == "__main__":
    main()
