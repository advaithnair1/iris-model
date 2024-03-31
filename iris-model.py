from numpy.random import seed
seed(8)
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from sklearn import datasets, model_selection 

from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, BatchNormalization
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from tensorflow.keras import regularizers

# 10% validation set size
iris_data = datasets.load_iris()
train_data, test_data, train_targets, test_targets = train_test_split(iris_data["data"], targets["target"], test_size = 0.1)

# Convert targets to a one-hot encoding
train_targets = tf.keras.utils.to_categorical(np.array(train_targets))
test_targets = tf.keras.utils.to_categorical(np.array(test_targets))

model = Sequential([
  Dense(64, activation = "relu", kernel_regularizer = regularizers.l2(weight_decay), input_shape = input_shape, kernel_initializer = 'he_uniform', bias_initializer = tf.keras.initializers.Constant(1.0)),
  Dense (128, kernel_regularizer = regularizers.l2(weight_decay), activation = "relu"), 
  Dense (128, kernel_regularizer = regularizers.l2(weight_decay), activation = "relu"),
  Dropout(dropout_rate),
  Dense (128, kernel_regularizer = regularizers.l2(weight_decay), activation = "relu"),
  Dense (128, kernel_regularizer = regularizers.l2(weight_decay), activation = "relu"),
  BatchNormalization(),
  Dense (64, kernel_regularizer = regularizers.l2(weight_decay), activation = "relu"),
  Dense (64, kernel_regularizer = regularizers.l2(weight_decay), activation = "relu"),
  Dropout(dropout_rate),
  Dense (64, kernel_regularizer = regularizers.l2(weight_decay), activation = "relu"),
  Dense (64, kernel_regularizer = regularizers.l2(weight_decay), activation = "relu"),
  Dense (3, activation = "softmax")
])

model.compile(
  optimizer = Adam(learning_rate = 0.0001), 
  loss = "categorical_crossentropy", 
  metrics = ["accuracy"]
)

history = model.fit(train_data, train_targets, epochs = 800, validation_split = 0.15, batch_size = 40)

try:
    plt.plot(reg_history.history['accuracy'])
    plt.plot(reg_history.history['val_accuracy'])
except KeyError:
    plt.plot(reg_history.history['acc'])
    plt.plot(reg_history.history['val_acc'])
plt.title('Accuracy vs. epochs')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(['Training', 'Validation'], loc='lower right')
plt.show()

plt.plot(reg_history.history['loss'])
plt.plot(reg_history.history['val_loss'])
plt.title('Loss vs. epochs')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(['Training', 'Validation'], loc='upper right')
plt.show() 
