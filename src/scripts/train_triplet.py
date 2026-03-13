from farm.train_triplet import train_triplet_model


model = train_triplet_model(
    npz_path="data/raw/mfc_features.npz",
    weights_path="artifacts/model.weights.h5",
    latent_dim=32,
    triplet_epochs=100,
    triplet_learning_rate=1e-4,
    batch_size=32,
    min_samples=2 * 32,
    apply_preprocessing=False,
    force_train=False,
)
