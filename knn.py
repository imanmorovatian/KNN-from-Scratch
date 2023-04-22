import numpy as np
from collections import Counter


class KNearestNeighbors:
    '''
    This class implement K Nearest Neighbour algorithm. It contain two main methods:
    1)fit and 2)predict
    '''

    def __init__(self, k, distance_metric='euclidean', weights="uniform"):

        self.k = k
        self.distance_metric = distance_metric
        self.weights = weights
        
        
    def fit(self, X, y):           
        
        self.X_train = np.array(X)       
        self.y_train = np.array(y)
        
        self.X_train_expanded = np.expand_dims(self.X_train, 1) # later, this will be used by predict function
        self.X_train_norm = np.sqrt(np.sum(self.X_train**2, axis=1)).reshape(-1,1) # later, this will be used by predict function
    
                
    def predict(self, X):

        '''This function is a helper one for predict function to compute Euclidean distance between train and test sets'''
        def euclidean(X_test):
            return np.sqrt(np.sum(((self.X_train_expanded - X_test)**2), axis=2))
    

        '''This function is a helper one for predict function to compute cosine distance between train and test sets'''
        def cosine(X_test):
            dot_product = np.multiply(self.X_train, np.transpose(X_test))
            X_test_norm = np.sqrt(np.sum(X_test**2, axis=1))
            return 1 - abs(dot_product / self.X_train_norm / X_test_norm)
    

        '''This function is a helper one for predict function to compute Manhattan distance between train and test sets'''
        def manhattan(X_test):
            return np.sum((self.X_train_expanded - X_test), axis=2)


        '''This function is a helper one for predict function to perform majority voting'''
        def majority_voting(votes):
            counter = Counter(votes)
            return counter.most_common(1)[0][0]
    

        '''This function is a helper one for predict function to perform majority voting'''
        def weighted_majority_voting(votes, weights):
            count = {}
            for vote, weight in zip(votes, weights):
                count.setdefault(vote, 0)
                count[vote] += weight
        
            return max(count.items(), key=lambda i: i[1])[0]
            

        X_test = np.array(X)    
        
        if self.distance_metric == 'euclidean':
            distances = euclidean(X_test)
        elif self.distance_metric == 'cosine':
            distances = cosine(X_test)
        elif self.distance_metric == 'manhattan':
            distances = manhattan(X_test)
            
        knn = np.transpose(np.argsort(distances, axis=0)[:self.k,:])
        knn_labels = self.y_train[knn]
        
        if self.weights == 'uniform':
            prediction = [majority_voting(knn_labels[i]) for i in range(len(knn))]
            
        elif self.weights == 'distance':
            weights = 1 / ( np.take_along_axis(distances, np.transpose(knn), axis=0)+1e-5 )
            weights = np.transpose(weights)
            prediction = [weighted_majority_voting(knn_labels[i], weights[i]) for i in range(len(knn))]
        
        return prediction





