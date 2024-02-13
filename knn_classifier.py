import numpy as np
import scipy.stats as stats
from os import sys
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)




#Define functions for calculating distances
def Euclidean(x1, x2):
    return np.sqrt(np.sum((np.array(x1) - np.array(x2))**2,axis=1))

def Manhattan(x, y):
    return np.sum(np.abs(np.array(x) - np.array(y)),axis=1)
            


#Define knn class
class KNNClassifier():
    def __init__(self, k=3, distance_metric='Euclidean'):
        self.k = k
        self.distance_metric = distance_metric


    #Define fit function
    def fit(self, X, y):
        self.X_train = X
        self.y_train = y

    #define predict function
    def predict(self, X):
        labels = np.zeros(len(X[:, 1])) #For saving labels
        for i in range(len(X[:, 1])):
            
            #These calculate distances between 8th dimension X[i] point and all points on X_train
            if self.distance_metric == 'Euclidean':
                distance = Euclidean(self.X_train, X[i].reshape(1,8))  
                
            elif self.distance_metric == 'Manhattan':
                distance = Manhattan(self.X_train, X[i].reshape(1,8))

            else:
                raise ValueError("No such a defined distance metric")
            
            k_indices = np.argsort(distance)[:self.k] #Gives indices of minimum distances
            k_classes = self.y_train[k_indices] #Gives labels of minimum distances
            mode = stats.mode(k_classes) #Which label is the most in neighbors?
            labels[i] = mode[0] # Predicted label for X[i] point
            

        return labels
 