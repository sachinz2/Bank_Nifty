import tensorflow as tf
from tensorflow.keras.layers import Input, Conv1D, BatchNormalization, ReLU, Dropout, LSTM, Dense
from tensorflow.keras.models import Model

def create_encoder(input_shape=(60, 12)):
    """
    Creates the 1D-CNN + LSTM encoder.

    Args:
        input_shape: The shape of the input data (timesteps, features).

    Returns:
        A Keras model representing the encoder.
    """
    inputs = Input(shape=input_shape)

    # 1D-CNN layers
    x = Conv1D(filters=32, kernel_size=3, strides=1, padding='same')(inputs)
    x = BatchNormalization()(x)
    x = ReLU()(x)
    x = Dropout(0.15)(x)

    x = Conv1D(filters=64, kernel_size=5, strides=1, padding='same')(x)
    x = BatchNormalization()(x)
    x = ReLU()(x)
    x = Dropout(0.15)(x)

    x = Conv1D(filters=64, kernel_size=7, strides=1, padding='same')(x)
    x = BatchNormalization()(x)
    x = ReLU()(x)
    x = Dropout(0.15)(x)

    # LSTM layer
    x = LSTM(128)(x)

    encoder = Model(inputs=inputs, outputs=x, name='encoder')
    return encoder

def create_pretrain_model(encoder):
    """
    Creates the supervised pre-training model.

    Args:
        encoder: The encoder model.

    Returns:
        A Keras model for supervised pre-training.
    """
    x = Dense(64, activation='relu')(encoder.output)
    outputs = Dense(1, activation='sigmoid')(x)

    pretrain_model = Model(inputs=encoder.input, outputs=outputs, name='pretrain_model')
    return pretrain_model

def create_actor_critic(encoder):
    """
    Creates the actor and critic models for PPO.

    Args:
        encoder: The encoder model.

    Returns:
        A tuple containing the actor and critic Keras models.
    """
    # Actor
    actor_x = Dense(64, activation='tanh')(encoder.output)
    actor_outputs = Dense(3, activation='softmax')(actor_x) # {Long, Cash, Short}
    actor = Model(inputs=encoder.input, outputs=actor_outputs, name='actor')

    # Critic
    critic_x = Dense(64, activation='relu')(encoder.output)
    critic_outputs = Dense(1)(critic_x) # Value scalar
    critic = Model(inputs=encoder.input, outputs=critic_outputs, name='critic')

    return actor, critic