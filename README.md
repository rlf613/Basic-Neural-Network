# Basic Neural Network

### v1.08

A simple Neural Network written in Python usually only NumPy.

#### TODO:

Add in a stratify feature in Split()

Includes:  
    - `Encoder` One-hot and/or class integers.  
    - `LoadModel` Loads a saved model's weights/biases to re-use.  
    - `MinMaxScaler` Scales data between 0 and 1 (normalisation).  
    - `Split` Splits the data into training and testing sets.  

Neural Network:  
    - Activations: `relu` `leaky_relu` `tanh` `sigmoid` `softmax`  
    - Loss Functions: `spare_categorical_crossentropy` `categorical_crossentropy` `mse` `mae`  
    - Optimizers: `adam` `rmsprop` `adadelta` `sgd`  

`nnet.py` contains the class for the Neural Network (NN) and other class functions.

##### 98% on MNIST Digits.

## Usage

#### main.py
```python
import numpy as np
from nnet import NN, Encoder, MinMaxScaler, Split, LoadModel
import matplotlib.pyplot as plt

# === MNIST HANDWRITTEN DIGITS ===
with np.load('datasets/mnist.npz') as data: X, Y = data['X'], data['Y']
# === PRE-PROCESSING ===
X = X.reshape(X.shape[0], -1)
X = MinMaxScaler.transform(X)
labels = Encoder()
Y = labels.encode(Y)
X_train, X_test, Y_train, Y_test = Split.split(X, Y)
# === NEURAL NETWORK ===
model = NN(verbose=True)
model.input(input_size=X_train.shape[1])
model.hidden(neurons=512, activation='relu', dropout=0.2)
model.hidden(neurons=512, activation='relu')
model.output(output_size=10, activation='softmax')
model.compile(loss='sparse_categorical_crossentropy')
model.train(X_train, Y_train, batch_size=128, epochs=15, valid_split=0.2)
model.evaluate(X_test, Y_test)
model.plot()
# === SAVE & LOAD ===
model.save('mnist')
mnist = LoadModel('mnist')
# === PLOT PREDICTION ===
rnum = np.random.randint(0, X.shape[0])
prediction, acc = model.predict(X[rnum])
print(f'Model: {labels.decode(prediction)} ({acc:.2%}) | Actual: {labels.decode(Y[rnum])}')
img_dims = int(np.sqrt(X.shape[1]))
plt.imshow(X[rnum].reshape(img_dims, img_dims), cmap='bone_r')
plt.show()
