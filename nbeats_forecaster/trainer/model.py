import logging

import tensorflow as tf


# Create NBeatsBlock custom layer
class NBeatsBlock(tf.keras.layers.Layer):
    def __init__(self, input_size: int, theta_size: int, horizon: int, n_neurons: int, n_layers: int, **kwargs):
        super().__init__(**kwargs)
        self.input_size = input_size
        self.theta_size = theta_size
        self.horizon = horizon
        self.n_neurons = n_neurons
        self.n_layers = n_layers

        # Block contains stack of 4 fully connected layers each has ReLU activation
        self.hidden = [tf.keras.layers.Dense(n_neurons, activation='relu') for _ in range(n_layers)]

        # Output of block is a theta layer with linear activation
        self.theta_layer = tf.keras.layers.Dense(theta_size, activation='linear', name='theta')

    def call(self, inputs):
        x = inputs
        for layer in self.hidden:
            x = layer(x)
        theta = self.theta_layer(x)

        # Output the backcast and the forecast from theta
        backcast, forecast = theta[:, :self.input_size], theta[:, -self.horizon:]

        return backcast, forecast


def build_nbeats_forecaster(hparams: dict) -> tf.keras.Model:
    """Define and compile N-BEATS forecasting model.

    Args:
        hparams (dict): Dictionary containing model training arguments.

    Returns:
        tf.keras.Model: A compiled TensorFlow model.
    """
    # 1. Setup an instance of NBeatsBlock
    nbeats_block_layer = NBeatsBlock(input_size=hparams['input_size'],
                                     theta_size=hparams['theta_size'],
                                     horizon=hparams['horizon'],
                                     n_neurons=hparams['n_neurons'],
                                     n_layers=hparams['n_layers'],
                                     name='InitialBlock')

    # 2. Create input to stack
    stack_input = tf.keras.layers.Input(shape=(hparams['input_size']), name='stack_input')

    # 3. Create initial backcast and forecast input (backwards prediction + horizon prediction)
    residuals, forecast = nbeats_block_layer(stack_input)

    # 4. Create stacks of block layers
    for i, _ in enumerate(range(hparams['n_stacks'] - 1)):  # first stack is already created in (3)

        # 5. Use the NBeatsBlock to calculate the backcast as well as the forecast
        backcast, block_forecast = NBeatsBlock(input_size=hparams['input_size'],
                                               theta_size=hparams['theta_size'],
                                               horizon=hparams['horizon'],
                                               n_neurons=hparams['n_neurons'],
                                               n_layers=hparams['n_layers'],
                                               name=f'NBeatsBlock_{i}')(residuals)  # pass in the residuals

        # 6. Create the double residual stacking
        residuals = tf.keras.layers.subtract([residuals, backcast], name=f'subtract_{i}')
        forecast = tf.keras.layers.add([forecast, block_forecast], name=f'add_{i}')

    # 7. Put the stack model together
    model = tf.keras.Model(inputs=stack_input, outputs=forecast, name='model_NBEATS')

    # 8. Compile the model with MAE loss
    model.compile(loss='mae', optimizer=tf.keras.optimizers.Adam())

    return model


def train_evaluate(
    hparams: dict,
    train_dataset: tf.data.Dataset,
    test_dataset: tf.data.Dataset
) -> tf.keras.callbacks.History:
    """Train and evaluate TensorFlow N-BEATS forecasting model.

    Args:
        hparams (dict): Dictionary containing model training arguments.
        train_dataset (tf.data.Dataset): Training dataset
        test_dataset (tf.data.Dataset): Testing Dataset

    Returns:
        tf.keras.callbacks.History: Keras callback that records training event history.
    """
    mirrored_strategy = tf.distribute.MirroredStrategy()
    with mirrored_strategy.scope():
        model = build_nbeats_forecaster(hparams=hparams)
        logging.info(model.summary())

    history = model.fit(x=train_dataset,
                        epochs=hparams['n_epochs'],
                        validation_data=test_dataset,
                        verbose=0,
                        callbacks=[tf.keras.callbacks.EarlyStopping(monitor='val_loss',
                                                                    patience=200,
                                                                    restore_best_weights=True),
                                   tf.keras.callbacks.ReduceLROnPlateau(monitor='val_loss',
                                                                        patience=100,
                                                                        verbose=1)])

    logging.info("Test MAE: %s", model.evaluate(test_dataset))

    # Export Keras model in TensorFlow SavedModel format.
    model.save(hparams['model-dir'])

    return history
