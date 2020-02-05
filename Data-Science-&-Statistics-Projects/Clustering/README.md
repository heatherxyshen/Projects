# Clustering-3-methods.R

These clustering methods will be applied to the faithful dataset found in R. It is a dataset with n = 272 observations. There are two variables in this dataset but we will focus on one variable – eruptions. 
It is the waiting time for the eruption. We will use it to create a new dataset consists of two variables: current waiting time and the next waiting time (in this new dataset, there is only n = 271 observations)

We will implement 3 clustering methods on this dataset

## Method 1
Implement the mean-shift algorithm to form clusters. Use Gaussian kernel with bandwidth h = 0.2.

## Method 2
Implement the k-means algorithm to form clusters with k = 4. We will use a random initialization (randomly choose 4 observations as the initial centers). 
We will also repeat the entire process at least 1000 times and choose the one that minimizes the k-means objective function to mitigate bad initialization points.

## Method 3
Implement the spectral clustering with a random walk Laplacian and k = 4. Use a Gaussian kernel to compute the kernel similarity matrix with h = 0.2

# Clustering-PCA-kmeans.py

## Exercise 1
We perform k-means clustering manually, with K = 2, on a small example with n = 8 observations and p = 2 features. The observations are as follows:
<p align="center">
<img src = "https://github.com/heatherxyshen/School/blob/master/Data-Science-%26-Statistics-Projects/Clustering/PCA-kmeans-Observations.png" width=300>
</p>

We first plot the observations. Next, we go through each step of a k-means clustering algorithm before running the complete algorithm on the data. Finally, we plot the resulting clusters.

## Exercise 2
Here, we generate simulated data and perform PCA and k-means clustering on it.

### Part a
Generate a simulated data set with 25 observations in each of three classes (i.e. 75 observations total), and 50 variables (features)

### Part b
Perform PCA on the 75 observations.

### Part c
Perform k-means clustering of the obervations with K = 3.

### Part d
Perform k-means clustering of the obervations with K = 2.

### Part e
Perform k-means clustering of the obervations with K = 4.

### Part f
Now perform k-means clustering with K = 3 on the first two principal component score vectors, rather than on the raw data.

### Part g
Perform k-means clustering with K = 3 on the data after scaling each variable to have standard deviation one.

## Exercise 3
Generate data with three features, with 30 data points in each of three classes as follows:
<p align="center">
<img src = "https://github.com/heatherxyshen/School/blob/master/Data-Science-%26-Statistics-Projects/Clustering/kmeans-classes.png" width = 300>
</p>

We have a three-dimensional dataset with 3 clusters lying near the surface of a sphere at (1, 0, 0), (0, 1, 0) and (0, 0, 1). Similar to before, we write a program to run k-means on these data and plot the resulting clusters.


