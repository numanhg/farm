import argparse
import os

import numpy as np
from scipy.spatial.distance import euclidean
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import LabelEncoder

from meine import TripletAutoencoder


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--prototype_label_id", type=int, default=0)
    return parser.parse_args()


def load_base_data():
    data_npz = np.load("../data/mfc_features_no_packed_timestamp_processed.npz")

    features = data_npz["features"]
    family_names = data_npz["family_names"]
    categories = data_npz["categories"]

    malicious_mask = categories == "malicious"
    evolving_mask = categories == "malicious-evolving"
    unseen_mask = categories == "malicious-unseen"

    le = LabelEncoder()
    le.fit(family_names)

    return {
        "malicious_features": features[malicious_mask],
        "evolving_features": features[evolving_mask],
        "unseen_features": features[unseen_mask],
        "malicious_labels": le.transform(family_names[malicious_mask]),
        "evolving_labels": le.transform(family_names[evolving_mask]),
        "unseen_labels": le.transform(family_names[unseen_mask]),
    }


def load_retrain_data(prototype_label_id: int):
    retrain_data = np.load(f"retrain_{prototype_label_id}.npz")
    return {
        "x_retrain": retrain_data["X_retrain"],
        "y_retrain": retrain_data["y_retrain"],
        "retrain_drifted_indices": retrain_data["retrain_drifted_indices"],
        "padded_prototype_indices": retrain_data["padded_protoype_indices"],
    }


def load_artifacts(prototype_label_id: int):
    thresholds = np.load(f"./thresholds_{prototype_label_id}.npz", allow_pickle=True)
    centroids = np.load(f"./centroids_{prototype_label_id}.npz", allow_pickle=True)
    epsilons = np.load(f"./epsilons_{prototype_label_id}.npz", allow_pickle=True)
    return thresholds, centroids, epsilons


def build_model(x_retrain: np.ndarray, y_retrain: np.ndarray, prototype_label_id: int) -> TripletAutoencoder:
    return TripletAutoencoder(
        X_train=x_retrain,
        y_train=y_retrain,
        triplet_weights_path=os.path.abspath(f"model_updated_{prototype_label_id}.weights.h5"),
        triplet_epochs=20,
    )


def evaluate_model(features: np.ndarray, padded_prototype_indices: np.ndarray, centroids, thresholds):
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
            padded_index = padded_prototype_indices[idx] if idx < len(padded_prototype_indices) else -1
            drifted_indices.append(padded_index)
        else:
            status.append("no-drift")

    return np.array(predictions), np.array(status), np.array(drifted_indices)


def build_eval_set(
    encoder_model,
    x_test: np.ndarray,
    y_test: np.ndarray,
    evolving_features: np.ndarray,
    evolving_labels: np.ndarray,
    prototype_label_id: int,
    retrain_drifted_indices: np.ndarray,
):
    label_mask = evolving_labels == prototype_label_id
    all_indices = np.arange(len(evolving_labels))
    exclude_mask = ~np.isin(all_indices, retrain_drifted_indices)
    final_mask = label_mask & exclude_mask

    x_proto = evolving_features[final_mask]
    x_proto = encoder_model.predict(x_proto)

    x_eval = np.vstack([x_test, x_proto])
    y_eval = np.hstack([y_test, evolving_labels[final_mask]])

    return x_eval, y_eval


def main() -> None:
    args = parse_args()
    prototype_label_id = args.prototype_label_id

    base = load_base_data()
    retrain = load_retrain_data(prototype_label_id)
    thresholds, centroids, _ = load_artifacts(prototype_label_id)

    triplet_model = build_model(retrain["x_retrain"], retrain["y_retrain"], prototype_label_id)

    encoder_model = triplet_model.encoder_model
    x_test = encoder_model.predict(triplet_model.X_test)
    y_test = triplet_model.y_test

    x_eval, y_eval = build_eval_set(
        encoder_model=encoder_model,
        x_test=x_test,
        y_test=y_test,
        evolving_features=base["evolving_features"],
        evolving_labels=base["evolving_labels"],
        prototype_label_id=prototype_label_id,
        retrain_drifted_indices=retrain["retrain_drifted_indices"],
    )

    predictions, _, _ = evaluate_model(
        x_eval,
        retrain["padded_prototype_indices"],
        centroids,
        thresholds,
    )

    predictions = np.array(predictions).astype(int)
    mask = y_eval == prototype_label_id
    print("After: ", accuracy_score(y_eval[mask], predictions[mask]))


if __name__ == "__main__":
    main()
