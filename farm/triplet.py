import os

from keras.callbacks import ModelCheckpoint
from keras.optimizers import Adam

from farm.meta_dicts import build_meta_dicts, save_meta_dicts
from farm.triplet_batch import triplet_generator
from farm.triplet_data import split_train_val_test
from farm.triplet_losses import make_triplet_loss
from farm.triplet_network import build_autoencoder, build_triplet_multitask_model


class TripletAutoencoder:
    def __init__(
        self,
        X_train,
        y_train,
        triplet_weights_path,
        triplet_initial_weights=None,
        save_meta_dict=False,
        triplet_epochs=30,
        latent_dim=32,
        triplet_loss_margin=50.0,
        batch_size=32,
        triplet_learning_rate=1e-4,
        triplet_loss_weights=(1.0, 0.4),
        dbscan_min_samples=None,
        force_train=False,
    ):
        self.triplet_weights_path = triplet_weights_path
        self.triplet_initial_weights = triplet_initial_weights
        self.force_train = force_train

        self.latent_dim = latent_dim
        self.triplet_loss_margin = triplet_loss_margin
        self.batch_size = batch_size
        self.triplet_learning_rate = triplet_learning_rate
        self.triplet_epochs = triplet_epochs
        self.triplet_loss_weights = triplet_loss_weights
        self.save_meta_dict = save_meta_dict
        self.dbscan_min_samples = dbscan_min_samples if dbscan_min_samples is not None else 2 * latent_dim

        (
            self.X_train,
            self.X_val,
            self.X_test,
            self.y_train,
            self.y_val,
            self.y_test,
        ) = split_train_val_test(X_train, y_train)

        self.input_shape = self.X_train.shape[1]
        self.autoencoder_model, self.encoder_model = build_autoencoder(
            input_shape=self.input_shape,
            latent_dim=self.latent_dim,
        )
        self.triplet_model = build_triplet_multitask_model(
            base_model=self.autoencoder_model,
            input_shape=self.input_shape,
        )

        self._compile_model()
        self._train_or_load_model()
        self.save_dict_to_npz()

    class CustomModelCheckpoint(ModelCheckpoint):
        def on_epoch_end(self, epoch, logs):
            current_loss = logs.get("val_loss")
            if current_loss == 0:
                if self.verbose > 0:
                    print(f"Epoch {epoch + 1}: loss is 0.0, skipping saving model")
                return
            super().on_epoch_end(epoch, logs)

    def _compile_model(self):
        self.triplet_model.compile(
            optimizer=Adam(self.triplet_learning_rate),
            loss=[make_triplet_loss(self.latent_dim, self.triplet_loss_margin), "mse"],
            loss_weights=[self.triplet_loss_weights[0], self.triplet_loss_weights[1]],
        )

    def _get_callbacks(self):
        return [
            self.CustomModelCheckpoint(
                filepath=self.triplet_weights_path,
                save_weights_only=True,
                save_best_only=True,
                monitor="val_loss",
                mode="min",
                verbose=3,
            )
        ]

    def _load_weights_if_available(self):
        if os.path.exists(self.triplet_weights_path) and not self.force_train:
            print("INFO: Loading weights from previous training")
            self.triplet_model.load_weights(self.triplet_weights_path)
            return True

        if self.triplet_initial_weights is not None:
            print("INFO: Loading initial weights from provided path")
            self.triplet_model.load_weights(self.triplet_initial_weights)

        return False

    def _train_model(self):
        print("INFO: Training with triplet loss")

        steps_per_epoch = max(1, len(self.X_train) // self.batch_size)
        validation_steps = max(1, len(self.X_val) // self.batch_size)

        self.triplet_model.fit(
            triplet_generator(self.X_train, self.y_train, self.batch_size, self.latent_dim),
            validation_data=triplet_generator(self.X_val, self.y_val, self.batch_size, self.latent_dim),
            validation_steps=validation_steps,
            steps_per_epoch=steps_per_epoch,
            epochs=self.triplet_epochs,
            callbacks=self._get_callbacks(),
        )

    def _train_or_load_model(self):
        loaded = self._load_weights_if_available()
        if not loaded:
            print("INFO: No reusable checkpoint found, training model")
            self._train_model()

    def save_dict_to_npz(self, save_dir=None):
        if not self.save_meta_dict:
            return

        self.thresholds_dict, self.centroids_dict, self.eps_dict = build_meta_dicts(
            encoder_model=self.encoder_model,
            X_train=self.X_train,
            y_train=self.y_train,
            X_val=self.X_val,
            y_val=self.y_val,
            min_samples=self.dbscan_min_samples,
        )
        save_meta_dicts(
            thresholds_dict=self.thresholds_dict,
            centroids_dict=self.centroids_dict,
            eps_dict=self.eps_dict,
            save_dir=save_dir,
        )
