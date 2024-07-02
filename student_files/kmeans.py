"""
File: kmeans.py
Project: Downloads
File Created: Feb 2021
Author: Rohit Das
"""

import numpy as np


class KMeans(object):
    def __init__(self, points, k, init="random", max_iters=10000, rel_tol=1e-05):
        """
        Args:
            points: NxD numpy array, where N is # points and D is the dimensionality
            K: number of clusters
            init : how to initial the centers
            max_iters: maximum number of iterations (Hint: You could change it when debugging)
            rel_tol: convergence criteria with respect to relative change of loss (number between 0 and 1)
        Return:
            none
        """
        self.points = points
        self.K = k
        if init == "random":
            self.centers = self.init_centers()
        else:
            self.centers = self.kmpp_init()
        self.assignments = None
        self.loss = 0.0
        self.rel_tol = rel_tol
        self.max_iters = max_iters

    def init_centers(self):
        """
            Initialize the centers randomly
        Return:
            self.centers : K x D numpy array, the centers.
        Hint: Please initialize centers by randomly sampling points from the dataset in case the autograder fails.
        """
        # figure out how big the array is - how many data points
        r = self.points.shape[0]
        # get k amount of random numbers up to r (array from 0-r, pick random indices)
        randos = np.random.choice(np.arange(r), self.K, replace=False)
        # get the points at the indices and return them
        self.centers = self.points[randos]
        return self.centers

        raise NotImplementedError

    def kmpp_init(self):
        """
            Use the intuition that points further away from each other will probably be better initial centers
        Return:
            self.centers : K x D numpy array, the centers.
        """
        raise NotImplementedError

    def update_assignment(self):
        """
            Update the membership of each point based on the closest center
        Return:
            self.assignments : numpy array of length N, the cluster assignment for each point
        Hint: Donot use loops for the update_assignment function
        Hint: You could call pairwise_dist() function
        Hint: In case the np.sqrt() function is giving an error in the pairwise_dist() function, you can use the squared distances directly for comparison.
        """
        #r, c, = arr.shape
        # get distance
        difference = pairwise_dist(self.points, self.centers)
        #difference = arr[:, None, :] - self.centers
        # get the distance from the means
        #disty = np.linalg.norm(difference, axis = -1)
        # get the closest ones and make the clusters
        clusters = np.argmin(difference, axis = -1)
        self.assignments = clusters
        return clusters
        #raise NotImplementedError

    def update_centers(self):
        """
            update the cluster centers
        Return:
            self.centers: new centers, a new K x D numpy array of float dtype, where K is the number of clusters, and D is the dimension.

        HINT: Points may be integer, but the centers should not have to be. Watch out for dtype casting!
        HINT: If there is an empty cluster then it won't have a cluster center, in that case the number of rows in self.centers can be less than K.
        """
        # new centers array
        # there should be k number of centers and c dimensions per center
        new_means = np.zeros((self.K, self.centers.shape[1]))
        for k in range(self.K):
            points = self.points[self.assignments == k]
            if len(points) > 0:
                new_means[k] = np.mean(points, axis=0)
        self.centers = new_means
        return self.centers
        #raise NotImplementedError

    def get_loss(self):
        """
            The loss will be defined as the sum of the squared distances between each point and it's respective center.
        Return:
            self.loss: a single float number, which is the objective function of KMeans.
        """
        loss = 0
        for k in range(self.K):
            # get only the points that belong to cluster k
            points = self.points[self.assignments == k]
            # get the center that belongs to k
            center = self.centers[k]

            if len(points) > 0:
                squared = np.sum(np.square(points - center))
                loss += squared
            # diff = points - center
            # squared_sum = np.sum(np.square(diff))
            # loss += squared_sum
        self.loss = loss
        return self.loss
        #raise NotImplementedError

    def train(self):
        """
            Train KMeans to cluster the data:
                0. Recall that centers have already been initialized in __init__
                1. Update the cluster assignment for each point
                2. Update the cluster centers based on the new assignments from Step 1
                3. Check to make sure there is no mean without a cluster,
                   i.e. no cluster center without any points assigned to it.
                   - In the event of a cluster with no points assigned,
                     pick a random point in the dataset to be the new center and
                     update your cluster assignment accordingly.
                4. Calculate the loss and check if the model has converged to break the loop early.
                   - The convergence criteria is measured by whether the percentage difference
                     in loss compared to the previous iteration is less than the given
                     relative tolerance threshold (self.rel_tol).
                   - Relative tolerance threshold (self.rel_tol) is a number between 0 and 1.
                5. Iterate through steps 1 to 4 max_iters times. Avoid infinite looping!

        Return:
            self.centers: K x D numpy array, the centers
            self.assignments: Nx1 int numpy array
            self.loss: final loss value of the objective function of KMeans.

        HINT: Donot loop over all the points in every iteration. This may result in time out errors
        HINT: Make sure to care of empty clusters. If there is an empty cluster the number of rows in self.centers can be less than K.
        """
        prev = float('inf')
        for iter in range(self.max_iters):
            # update the cluster assignment for each point
            self.update_assignment()
            self.update_centers()
            #check to make sure there's no mean without a cluster
            if len(self.centers) < self.K :
                missing = self.K - len(self.centers)
                # if so, then pick a random point in the dataset to be the new center
                #r = self.points.shape[0]
                # get k amount of random numbers up to r (array from 0-r, pick random indices)
                randos = np.random.choice(self.points.shape[0], missing, replace=False)
                new = self.points[randos]
                # get the points at the indices and return them
                self.centers = np.concatendate((self.centers, new), axis = 0)
            # calculate the loss and check if the model has converged to break the loop early
            self.loss =self.get_loss()
            if abs(prev - self.loss) / prev < self.rel_tol:
                break
            prev = self.loss
        return self.centers, self.assignments, self.loss
        #raise NotImplementedError


def pairwise_dist(x, y):
    """
    Args:
        x: N x D numpy array
        y: M x D numpy array
    Return:
            dist: N x M array, where dist2[i, j] is the euclidean distance between
            x[i, :] and y[j, :]

    HINT: Do not use loops for the pairwise_distance function
    """
    x_square = np.sum(np.square(x), axis=1, keepdims=True)
    y_square = np.sum(np.square(y), axis = 1)
    xy = np.dot(x, y.T)
    disty = np.sqrt(x_square + y_square - 2 * xy)
    return disty
    #raise NotImplementedError


def rand_statistic(xGroundTruth, xPredicted):
    """
    Args:
        xPredicted : list of length N where N = no. of test samples
        xGroundTruth: list of length N where N = no. of test samples
    Return:
        Rand statistic value: final coefficient value as a float

    HINT: You can use loops for this function.
    HINT: The idea is to make the comparison of Predicted and Ground truth in pairs.
        1. Choose a pair of points from the Prediction.
        2. Compare the prediction pair pattern with the ground truth pair.
        3. Based on the analysis, we can figure out whether it's a TP/FP/FN/FP.
        4. Then calculate rand statistic value
    """
    TP = 0
    FP = 0
    TN = 0
    FN = 0
    # rand = (TP + TN)/(TP + FP + FN + TN)
    for i in range(len(xGroundTruth)):
        for j in range(i + 1, len(xGroundTruth)):
            ground = (xGroundTruth[i], xGroundTruth[j])
            predicted = (xPredicted[i], xPredicted[j])
            # TP
            if ground == predicted:
                TP += 1
            # false negative
            elif ground[0] == ground[1] and predicted[0] != predicted[1]:
                FN += 1
            # false positive
            elif ground[0] != ground[1] and predicted[0] == predicted[1]:
                FP += 1
            else:
                TN += 1
    randStat = float(TP + TN) / float(TP + FP + FN + TN)
    # 1. choose a pair of points from the prediction

    # 2. Compare the prediction pair pattern with the ground truth pair

    # 3. Based on the analysis, we can figure out if it's TP/FP/TN/FN

    # 4. Calculate the rand stat
    return randStat
    #raise NotImplementedError
