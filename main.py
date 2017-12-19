from mnist import MNIST
import numpy as np

mndata = MNIST('samples')

images, labels = mndata.load_training()

index = np.random.randint(0, len(images))  # choose an index ;-)
print(mndata.display(images[index]))
