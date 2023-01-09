# KNN from Scratch Using Numpy
K-Nearest Neighbors (KNN) is a supervised method used for classification and regression. There are optimized implemention of this algorithm in different libraries such as scikit-learn, I implement knn for classification in this repository from scratch in order to understand what is happening behind KNN algorithm. Classification for a point not assigned a label using KNN is done in the following step
1. K nearest neighbors of that point are detetermined
2. The label for which most neighbors vote will be assign to that point

To determine nearest neighbors, different metrics can be used. Euclidean, Manhattan, and cosine are implemented in this repository.
An idea can improve performance of KNN is that all neighbors of a point do not affect that point in a same way, and neighbors which are far from the point have less impact compared to neighbors which are closer to the point. This idea is implemented, and "weights" argument in predict method controls that. If this argument is set to "distance", the far the neighbor, the less its effect.