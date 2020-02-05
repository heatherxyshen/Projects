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
