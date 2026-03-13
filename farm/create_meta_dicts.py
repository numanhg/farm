from farm.meta_dicts import build_meta_dicts, save_meta_dicts


def create_meta_dictionaries(
    model,
    output_dir,
    min_samples=None,
    log_clusters=True,
):
    latent_dim = model.latent_dim
    effective_min_samples = min_samples if min_samples is not None else 2 * latent_dim

    thresholds_dict, centroids_dict, eps_dict = build_meta_dicts(
        encoder_model=model.encoder_model,
        X_train=model.X_train,
        y_train=model.y_train,
        X_val=model.X_val,
        y_val=model.y_val,
        min_samples=effective_min_samples,
        log_clusters=log_clusters,
    )
    save_meta_dicts(
        thresholds_dict=thresholds_dict,
        centroids_dict=centroids_dict,
        eps_dict=eps_dict,
        save_dir=output_dir,
    )

    return thresholds_dict, centroids_dict, eps_dict
