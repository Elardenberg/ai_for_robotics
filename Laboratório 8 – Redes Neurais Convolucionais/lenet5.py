from tensorflow.keras import layers, activations
from tensorflow.keras.models import Sequential
import os


# Comment this line to enable training using your GPU
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

def make_lenet5():
    model = Sequential()

    # Todo: implement LeNet-5 model
    # Entry Layer
    model.add(layers.Conv2D(filters=1, kernel_size=(1, 1), strides=(1, 1), input_shape=(32, 32, 1)))

    # Layer 1 - Conv2D
    # model.add(layers.Conv2D(filters=nf, kernel_size=(fx, fy), strides=(sx, sy), activation=activations.fun))
    model.add(layers.Conv2D(filters=6, kernel_size=(5, 5), strides=(1, 1), activation=activations.tanh))

    # Layer 2 - AveragePooling2D
    # model.add(layers.AveragePooling2D(pool_size=(px, py), strides=(sx, sy)))
    model.add(layers.AveragePooling2D(pool_size=(2, 2), strides=(2, 2)))

    # Following layers
    model.add(layers.Conv2D(filters=16, kernel_size=(5, 5), strides=(1, 1), activation=activations.tanh))
    model.add(layers.AveragePooling2D(pool_size=(2, 2), strides=(2, 2)))


    model.add(layers.Flatten())
    model.add(layers.Dense(120, activation=activations.tanh))
    model.add(layers.Dense(84, activation=activations.tanh))
    model.add(layers.Dense(10, activation=activations.softmax))


    return model
