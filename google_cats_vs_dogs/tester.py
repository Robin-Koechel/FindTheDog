import os
import numpy as np
import tensorflow as tf
from tensorflow import keras
import pandas as pd
import matplotlib.pyplot as plt

base_dir = "D:\\pycharm\\LearnersLearners\\cats_and_dogs_filtered"
base_model = "./part3b_adopted_model_tf"
base_history_csv = "./part3b_adopted_training.log"

validation_dir = os.path.join(base_dir, 'validation')
validation_cats_dir = os.path.join(validation_dir, 'cats')
validation_dogs_dir = os.path.join(validation_dir, 'dogs')

model = keras.models.load_model(base_model)

# panic prints - is the model really there? :)
# print(model.get_weights())
# print(model.get_config())

# load history off csv
log_data = pd.read_csv(base_history_csv, sep=',', engine='python')

acc = log_data['acc']
val_acc = log_data['val_acc']
loss = log_data['loss']
val_loss = log_data['val_loss']

# Get number of epochs
epochs = range(len(acc))
plt.plot(epochs, acc)
plt.plot(epochs, val_acc)
plt.title('Training and validation accuracy')
plt.figure()
plt.plot(epochs, loss)
plt.plot(epochs, val_loss)
plt.title('Training and validation loss')
plt.show()

