# Import necessary libraries
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os

# Set Keras backend to torch
os.environ["KERAS_BACKEND"] = "torch"
import keras
from keras.datasets import mnist

# import functions to load and preprocess the data 
from neural_network.preprocess import preprocess_data

X_train, y_train, X_test, y_test, y_train_encoded, y_test_encoded = preprocess_data()

# Create the model
model = keras.Sequential([
    keras.layers.Input(shape=(784,)),
    keras.layers.Dense(32, activation='relu'),
    keras.layers.Dense(10, activation='softmax')
])

# Print model summary
print(model.summary())

# Compile the model
epochs=150
model.compile(
    optimizer=keras.optimizers.SGD(learning_rate=0.00020),
    loss="categorical_crossentropy",
    metrics=["accuracy"]
)

history= model.fit(
    X_train,
    y_train_encoded,
    batch_size=X_train.shape[1],
    epochs=epochs,
    shuffle=False,                
    validation_data=(X_test, y_test_encoded),
    verbose=2
)

# Accuracy/Loss plot
accuracy, val_accuracy = history.history['accuracy'], history.history['val_accuracy']
loss, val_loss = history.history['loss'], history.history['val_loss']
epochs_trained = range(len(accuracy))

plt.figure(figsize=(15, 15))
plt.subplot(2, 2, 1)
plt.plot(epochs_trained, accuracy, label='Training Accuracy')
plt.plot(epochs_trained, val_accuracy, label='Validation Accuracy')
plt.legend(loc='lower right')
plt.title('Training and Validation Accuracy')
plt.subplot(2, 2, 2)
plt.plot(epochs_trained, loss, label='Training Loss')
plt.plot(epochs_trained, val_loss, label='Validation Loss')
plt.legend(loc='upper right')
plt.title('Training and Validation Loss')
plt.savefig('results/accuracy_loss.png')

# Plot confusion matrix - literally "the model confuses letter X, with what"
from sklearn.metrics import confusion_matrix
y_pred = np.argmax(model.predict(X_test), axis=1)
plt.figure(figsize=(12, 12))
sns.heatmap(confusion_matrix(y_test, y_pred), cmap='viridis', annot=True, fmt='g')
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Confusion Matrix")
plt.savefig('results/confusion.png')