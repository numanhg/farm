from keras.layers import Activation, BatchNormalization, Dense, Dropout, Input, concatenate
from keras.models import Model


def dense_block(x, units, dropout_rate=0.2, activation="relu"):
    x = Dense(units)(x)
    x = BatchNormalization()(x)
    x = Activation(activation)(x)
    x = Dropout(dropout_rate)(x)
    return x


def build_autoencoder(input_shape, latent_dim):
    inputs = Input(shape=(input_shape,))

    x = dense_block(inputs, 1024)
    x = dense_block(x, 512)
    x = dense_block(x, 256)
    x = dense_block(x, 128)

    embedding_output = Dense(latent_dim)(x)

    x = dense_block(embedding_output, 128)
    x = dense_block(x, 256)
    x = dense_block(x, 512)
    x = dense_block(x, 1024)

    mse_output = Dense(input_shape, activation="sigmoid")(x)

    autoencoder_model = Model(inputs, [embedding_output, mse_output])
    encoder_model = Model(inputs, embedding_output)
    return autoencoder_model, encoder_model


def build_triplet_multitask_model(base_model, input_shape):
    input_anchor = Input(shape=(input_shape,))
    input_positive = Input(shape=(input_shape,))
    input_negative = Input(shape=(input_shape,))

    processed_anchor, mse_embeddings = base_model(input_anchor)
    processed_positive, _ = base_model(input_positive)
    processed_negative, _ = base_model(input_negative)

    triplet_embeddings = concatenate([processed_anchor, processed_positive, processed_negative])

    return Model(
        inputs=[input_anchor, input_positive, input_negative],
        outputs=[triplet_embeddings, mse_embeddings],
    )
