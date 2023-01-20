import argparse, os
import numpy as np
import pickle

import tensorflow as tf
import tensorflow.keras as keras
from tensorflow.keras import backend as K
from tensorflow.keras import Input
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.utils import multi_gpu_model

def autoencoder_model(input_dims):
    #Część enkoder
    inputLayer = Input(shape=(input_dims,))
    encoder = Dense(128, activation="relu")(inputLayer)
    encoder = Dense(64, activation="relu")(encoder)
    encoder = Dense(32, activation="relu")(encoder)
    encoder = Dense(8, activation="relu")(encoder)
    
    #Część dekoder
    decoder = Dense(32, activation="relu")(encoder)
    decoder = Dense(64, activation="relu")(decoder)
    decoder = Dense(128, activation="relu")(decoder)
    decoder = Dense(input_dims, activation=None)(decoder)

    return Model(inputs=inputLayer, outputs=decoder)


# epochs = 30
# n_mels = 128
def train(training_dir, model_dir, n_mels, frame, lr, batch_size, epochs):
    # Ładowanie danych
    train_data_file = os.path.join(training_dir, 'autoenkoder_data.pkl')
    with open(train_data_file, 'rb') as f:
        train_data = pickle.load(f) 
    
    #Przygotowanie modelu
    model = autoencoder_model(n_mels * frame)
    print(model.summary())
       
    #Kompilacja modelu
    model.compile(
        loss='mean_squared_error',
        optimizer=Adam(learning_rate=lr),
        metrics=['accuracy']
    )
    
    # Trenowanie modelu - te same dane używane do walidacji
    history = model.fit(
        train_data, 
        train_data,
        batch_size=batch_size,
        validation_split=0.1,
        epochs=epochs,
        shuffle=True,
        verbose=2
    )
    
    # Zapis
    os.makedirs(os.path.join(model_dir, 'model/1'), exist_ok=True)
    model.save(os.path.join(model_dir, 'model/1'))

    
def parse_arguments():
    parser = argparse.ArgumentParser()
    
    parser.add_argument('--epochs', type=int, default=10)
    parser.add_argument('--n_mels', type=int, default=128)
    parser.add_argument('--frame', type=int, default=5)
    parser.add_argument('--learning-rate', type=float, default=0.01)
    parser.add_argument('--batch-size', type=int, default=128)
    parser.add_argument('--model-dir', type=str, default=os.environ['SM_MODEL_DIR'])
    parser.add_argument('--training', type=str, default=os.environ['SM_CHANNEL_TRAINING'])

    args, _ = parser.parse_known_args()
    
    return args

if __name__ == '__main__':
    tf.random.set_seed(508)
    args = parse_arguments()
    epochs       = args.epochs
    n_mels       = args.n_mels
    frame        = args.frame
    lr           = args.learning_rate
    batch_size   = args.batch_size
    model_dir    = args.model_dir
    training_dir = args.training
    
    train(training_dir, model_dir, n_mels, frame, lr, batch_size, epochs)