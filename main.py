import random

import mlflow
import numpy as np
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from tensorflow.keras.datasets import mnist
from tensorflow.keras.layers import Dense, Activation
from tensorflow.keras.models import Sequential
from tensorflow.keras.optimizers import SGD
from tensorflow.keras.utils import to_categorical


def get_hyperparams(choices: dict) -> dict:
    """
    Returns a set of hyper-parameters to try.

    :param {dict} choices - A dict of choices to try
    :return {dict} A config
    """
    config = {}
    for key, val in choices.items():
        if isinstance(val, list):
            config[key] = random.choice(val)
        if isinstance(val, tuple):
            if isinstance(val[0], int):
                config[key] = random.randint(*val)
            else:
                config[key] = random.random() * val[1] + (val[1] - val[0])

    return config


def preprocess_data(preprocessor: str, X_train: np.ndarray, X_test: np.ndarray) -> tuple:
    """
    Preprocesses the input variables according to the specified preprocessor.
    Currently only accepts 'minmax'; other values will default to StandardScaler

    :param {str} preprocessor - 'minmax' or any other string.
    :param {np.ndarray} X_train - training data
    :param {np.ndarray} X_test - test data

    :return {tuple} X_train, X_test
    """
    if preprocessor == 'minmax':
        scaler = MinMaxScaler()
    else:
        scaler = StandardScaler()

    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    return X_train, X_test


def get_lr(lr_scheme: str, X_train: np.ndarray) -> float:
    """
    Gets a learning rate for SGD based on the scheme.

    :param {str} lr_scheme - One of 'lipschitz' or 'const'
    :param {np.ndarray} X_train - training data

    :return {float} learning rate
    """
    if lr_scheme == 'const':
        return 0.1

    m = X_train.shape[0]
    k = 10
    L = (k - 1) / (k * m) * np.linalg.norm(X_train)
    return 1. / L


def get_model(n_layers: int, n_units: int, initializer: str, lr: float) -> Sequential:
    """
    Builds a tf.keras model.

    :param {int} n_layers - number of layers
    :param {int} n_units - number of units/layer
    :param {str} initializer - one of 'glorot_uniform' or 'const'
    :param {float} lr - the learning rate

    :return {Sequential} a built model
    """
    model = Sequential()
    model.add(Dense(n_units, input_shape=(784,), name='layer1',
              kernel_initializer=initializer, bias_initializer=initializer))

    for i in range(n_layers - 2):
        model.add(Activation('relu'))
        model.add(Dense(
            n_units, name=f'layer{i+2}', kernel_initializer=initializer, bias_initializer=initializer))

    if n_layers >= 2:
        model.add(Activation('relu'))
        model.add(Dense(10, name=f'layer{n_layers}',
                  kernel_initializer=initializer, bias_initializer=initializer))

    model.add(Activation('softmax'))

    sgd = SGD(lr=lr)
    model.compile(loss='categorical_crossentropy',
                  optimizer=sgd, metrics=['accuracy'])
    return model


def fit_model(model: Sequential, n_epochs: int, X_train: np.ndarray, X_test: np.ndarray, y_train: np.ndarray, y_test: np.ndarray) -> float:
    """
    Fits a model.

    :param {Sequential} model - a compiled model
    :param {int} n_epochs - number of epochs
    :param {np.ndarray} X_train - training data
    :param {np.ndarray} X_test - test data
    :param {np.ndarray} y_train - training labels
    :param {np.ndarray} y_test - test labels

    :return {float} the validation accuracy
    """
    history = model.fit(X_train, y_train, epochs=n_epochs,
                        batch_size=128, validation_data=(X_test, y_test), verbose=1)

    return history.history['val_accuracy'][-1]


def run_experiment(X_train, X_test, y_train, y_test, config) -> float:
    """
    Runs one experiment.

    :param {np.ndarray} X_train - training data
    :param {np.ndarray} X_test - test data
    :param {np.ndarray} y_train - training labels
    :param {np.ndarray} y_test - test labels
    :param {dict} config - the configuration to run

    :return {float} the validation accuracy
    """
    # Preprocess data
    preprocessor = config['preprocessor']
    X_train, X_test = preprocess_data(preprocessor, X_train, X_test)

    # Get LR
    lr_scheme = config['lr_scheme']
    lr = get_lr(lr_scheme, X_train)
    print('LR =', lr)

    # Get initializer
    initializer = config['initializer']

    # Get model
    n_layers = config['n_layers']
    n_units = config['n_units']
    model = get_model(n_layers, n_units, initializer, lr)
    model.summary()

    # Fit model
    return fit_model(model, 100, X_train, X_test, y_train, y_test)


if __name__ == '__main__':
    # Get the data
    (X_train, y_train), (X_test, y_test) = mnist.load_data()
    X_train = X_train.reshape(60000, 784)
    X_test = X_test.reshape(10000, 784)

    X_train = X_train.astype('float32')
    X_test = X_test.astype('float32')

    y_train = to_categorical(y_train, 10)
    y_test = to_categorical(y_test, 10)

    choices = {
        'preprocessor': ['minmax', 'standard'],
        'lr_scheme': ['standard', 'lipschitz'],
        'n_layers': (1, 4),
        'n_units': [5, 10, 20],
        'initializer': ['zeros', 'glorot_uniform']
    }

    # Create an experiment
    experiment_id = mlflow.create_experiment('mnist')
    experiment = mlflow.get_experiment(experiment_id)
    print('Running experiment:', experiment.name)
    print('Artifact location:', experiment.artifact_location)

    # Set up best config tracking
    best_acc = 0.
    best_config = None
    for _ in range(2):
        with mlflow.start_run(experiment_id=experiment_id):
            # Get the config
            config = get_hyperparams(choices)

            # Log parameters
            for param, val in config.items():
                mlflow.log_param(param, val)

            val_acc = run_experiment(X_train, X_test, y_train, y_test, config)
            mlflow.log_metric('val_acc', val_acc)

            if val_acc > best_acc:
                best_acc = val_acc
                best_config = config

    print('Best config:', best_config)
    print('Best score:', best_acc)
