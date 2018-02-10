#import mnist
#import numpy as np
#
#def sigmoid(x):
#    return 1 / (1 + np.exp(-x))
#
#def deriv_sigmoid(x):
#    return sigmoid(x)*(1-sigmoid(x))
#
#mndata = mnist.MNIST('samples')
#
#images, labels = mndata.load_training()
#
#index = np.random.randint(0, len(images))  # choose an index ;-)
#print(mndata.display(images[index]))

import mnist
import matplotlib.pyplot as plt
import numpy as np

# Définition de la sigmoid
def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def deriv_sigmoid(x):
    return sigmoid(x)*(1-sigmoid(x))

# Importation des images
train_images = mnist.train_images()
train_labels = mnist.train_labels()

test_images = mnist.test_images()
test_labels = mnist.test_labels()

#plt.imshow(train_images[576,:,:], cmap='gray')
#plt.show()


# Taille de l'échantillon
siz = np.shape(train_images)
# Définition la matrice de poid
w = 2*np.random.random((10,siz[1]*siz[2])) - 1

for i in range(0,siz[0]):
    # Input de limage dans les entrée du réseau neuronal
    X0 = train_images[i,:,:]
    X0 = X0.flatten()
    
    # Calcul des influence des points avec la sigmoid et la matrice des poids
    Result = sigmoid(w.dot(X0))
    
    # Établissement du résultat correct [0,0,0,0,1,0,0,0]
    err = np.zeros((1,10))
    err[0,int(train_labels[i])] = 1 
    
    # Évaluation de la correctitude du résultat
    delta = (Result-err)*deriv_sigmoid(Result)
    
    # Ajustement des poids
    Ajus = delta.dot(w)
    Ajus2 = [Ajus for a in range(0,10)]
    Ajus2=np.squeeze(Ajus2)
    w += Ajus2
