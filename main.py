from mnist import MNIST
import numpy as np

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def deriv_sigmoid(x):
    return sigmoid(x)*(1-sigmoid(x))

mndata = MNIST('samples')

images, labels = mndata.load_training()

index = np.random.randint(0, len(images))  # choose an index ;-)
print(mndata.display(images[index]))
