import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from scipy.spatial import distance_matrix
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import scale
from mpl_toolkits import mplot3d


# import statistics as st
# import random

## PROBLEM 1
# intialise data
data = np.array([[1, 1, 0, 0.5, 4, 6, 5, 5.5], 
                 [3, 3.5, 4, 4.2, 1, 0.5, 0, 1.2]])

# Create DataFrame
df = pd.DataFrame({'X1': data[0], 'X2': data[1]})
 
# Create DataFrame
 
# Print the output.
print(df)

## Problem 1.a
# Plot Observations
plt.plot(df['X1'], df['X2'], 'ro')
plt.ylabel('X2')
plt.xlabel('X1')
plt.title('Plot of Observations')
plt.show()

## Problem 1.b
# Randomly assign clusters.  Report the cluster labels for each observation.
np.random.seed(193)
clust = np.random.choice(2, 8)
print(clust)

## Problem 1.c
# Centroid calculation
cent1 = df.loc[clust == 1, ].mean(axis = 0).values

cent0 = df.loc[clust == 0, ].mean(axis = 0).values

## Problem 1.d
# Assign new centroids
for index, row in df.iterrows():
    dist_0 = np.linalg.norm(row.values - cent0)
    dist_1 = np.linalg.norm(row.values - cent1)
    clust[index] = 0 if dist_0 < dist_1 else  1
    
print(clust)

# printcluster was only used to show how clusters evolved in Problem 3
def printcluster(data, cluster):
    names = np.array(['One', 'Two', 'Three'])
    ax = plt.axes(projection='3d')
    for i, col in enumerate(('blue', 'red', 'green')):
        ax.scatter3D(data[cluster == i][0], data[cluster == i][1],
                     data[cluster == i][2], label=names[i], c=col)
    plt.title('K-Means 3 Clusters')
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_zlabel('z')
    ax.view_init(40, 80)
    plt.show()

## Problem 1.e
# Repeating previous steps
def kmeans(data, k, iter = 1000):
    # calculate dimensions of variables (our problem is R^2)
    dim = len(data.columns)
    min_obj = float("inf")
    for n in np.arange(iter):
        # randomly assign clusters
        clust = np.random.choice(k, len(data))
    
        # Initial centroids
        cents = np.ones((k,dim))
        # printcluster(data, clust)

        # Initialize distance measure for change in centroids
        dist_change = np.ones(k)
        # While loop to run until centroids converge
        while(dist_change.max() > 1e-4):
            # Initialize new centroid
            new_cents = np.zeros((k,dim))
            # Calculate new centroids
            for i in np.arange(k):
                # If cluster has points, calculate new centroid
                # Else keep previous centroid
                if any(clust == i):
                    new_cents[i,:] = data.loc[clust == i,
                                              :].mean(axis = 0).values
    
            # Calculate distance from each point to new centroids
            dist = distance_matrix(data, new_cents)
            # Assigns new cluster based on closest new centroid
            clust = dist.argmin(axis=1)
            
            # printcluster(data, clust)
            
            # See how much centroids changed (for while loop)
            dist_change = distance_matrix(cents, new_cents).diagonal()
            
            # Keep centroids for next iteration
            cents = new_cents
            
        # Want to find centroids that minimize k-means objective function
        # Store centroids that have so far minimized obj function
        obj = np.mean(dist.min(axis = 1)**2)
        
        # If new one is better, replace previously best centroids
        # and store new minimum objective value
        if min_obj > obj:
            min_obj = obj
            final_cents = cents
    # Calculate distance from each point to final centroids
    dist = distance_matrix(data, final_cents)
    # Assigns new cluster based on closest new centroid
    clust = dist.argmin(axis=1)
 
    # Create new column for cluster assignment
    data['Cluster'] = clust
    # Return original data plus cluster assignment
    return(data)


km_df = kmeans(df.copy(), 2)

groups = km_df.groupby('Cluster')

## Problem 1.f
# Plot
fig, ax = plt.subplots()
for name, group in groups:
    ax.plot(group.X1, group.X2, marker='o', linestyle='', ms=8, label=name)
ax.legend()
plt.ylabel('X2')
plt.xlabel('X1')
plt.title('Cluster Assignments of Observations')
plt.show()


## PROBLEM 2
## Problem 2.a
# Generate data
def gendat(n, features, classes):
    dat = np.zeros((n, features))
    sizes = int(n/classes)
    for col in np.arange(features):
        class_means = np.random.random_integers(0, 
                                                n*2/classes, size = classes)
        dat[:,col] = np.concatenate((
            [np.random.normal(i, np.random.uniform(0, 7, size = 1), 
                              size = sizes) for i in class_means]), 
            axis = None)
        
    return(dat)
    
np.random.seed(3910)
data = gendat(75, 50, 3)
clust = np.concatenate([[0]*25, [1]*25, [2]*25], axis = None)
names = np.array(['One', 'Two', 'Three'])

for i, col in enumerate(('blue', 'red', 'green')):
    plt.scatter(data[clust==i, 0], data[clust==i, 1], label=names[i], c=col)
plt.xlabel('Feature 1')
plt.ylabel('Feature 2')
plt.title('Plot of Observations, 2 Features')
plt.legend(loc = 'best')
plt.show()

## Problem 2.b
# PCA 
pca = PCA(n_components=2)
data_std = StandardScaler().fit_transform(data)
data_pca = pca.fit_transform(data_std)

for i, col in enumerate(('blue', 'red', 'green')):
    plt.scatter(data_pca[clust==i, 0], data_pca[clust==i, 1], 
                label=names[i], c=col)
plt.xlabel('Principal Component 1')
plt.ylabel('Principal Component 2')
plt.title('Plot of First 2 Principal Components')
plt.legend(loc='best')
plt.show()

## Problem 2.c
# kmeans 3 clusters
km_3 = kmeans(pd.DataFrame(data.copy()), 3)
names = np.array(['One', 'Two', 'Three'])
for i, col in enumerate(('blue', 'red', 'green')):
    plt.scatter(km_3[km_3['Cluster'] == i][0], 
                km_3[km_3['Cluster'] == i][1], label=names[i], c=col)
plt.xlabel('Feature 1')
plt.ylabel('Feature 2')
plt.title('K-means 3 Clusters')
plt.legend(loc='best')
plt.show()

## Problem 2.d
# kmeans 2 clusters
km_2 = kmeans(pd.DataFrame(data.copy()), 2)
names = np.array(['One', 'Two'])
for i, col in enumerate(('blue', 'red')):
    plt.scatter(km_2[km_2['Cluster'] == i][0], 
                km_2[km_2['Cluster'] == i][1], label=names[i], c=col)
plt.xlabel('Feature 1')
plt.ylabel('Feature 2')
plt.title('K-means 2 Clusters')
plt.legend(loc='best')
plt.show()

## Problem 2.e
# kmeans 4 clusters
km_4 = kmeans(pd.DataFrame(data.copy()), 4)
names = np.array(['One', 'Two', 'Three', 'Four'])
for i, col in enumerate(('red', 'blue', 'green', 'orange')):
    plt.scatter(km_4[km_4['Cluster'] == i][0], 
                km_4[km_4['Cluster'] == i][1], label=names[i], c=col)
plt.xlabel('Feature 1')
plt.ylabel('Feature 2')
plt.title('K-means 4 Clusters')
plt.legend(loc='best')
plt.show()

## Problem 2.f
# kmeans on 2 pca components
km_pca = kmeans(pd.DataFrame(data_pca.copy()), 3)
names = np.array(['One', 'Two', 'Three'])
for i, col in enumerate(('blue', 'red', 'green')):
    plt.scatter(km_pca[km_pca['Cluster'] == i][0], 
                km_pca[km_pca['Cluster'] == i][1], label=names[i], c=col)
plt.xlabel('Principal Component 1')
plt.ylabel('Principal Component 2')
plt.title('K-means 3 Clusters, PCA')
plt.legend(loc='best')
plt.show()

## Problem 2.g
# scale data
scaled_data = scale(data, with_mean = False)
km_scaled = kmeans(pd.DataFrame(scaled_data.copy()), 3)
names = np.array(['One', 'Two', 'Three'])
for i, col in enumerate(('blue', 'red', 'green')):
    plt.scatter(km_scaled[km_scaled['Cluster'] == i][0], 
                km_scaled[km_scaled['Cluster'] == i][1], 
                label=names[i], c=col)
plt.xlabel('Feature 1')
plt.ylabel('Feature 2')
plt.title('K-means 3 Clusters, scaled data')
plt.legend(loc='best')
plt.show()



## PROBLEM 3
## Problem 3.a
class_size = 30
# making function for random normal distribution number generator
def wvar(classsize):
    var = (np.random.normal(loc = 0, scale = 0.5, 
                            size = classsize)).reshape((class_size, 1))
    return(var)

# set seed for reproducable results
np.random.seed(1420)

# class 1
theta_1 = np.random.uniform(low = -np.pi/8, high = np.pi/8, 
                            size = class_size).reshape((class_size, 1))
phi_1 = np.random.uniform(low = 0, high = 2 * np.pi, 
                          size = class_size).reshape((class_size, 1))
x_1 = np.add(np.multiply(np.sin(theta_1), np.sin(phi_1)), wvar(class_size))
y_1 = np.add(np.multiply(np.sin(theta_1), np.sin(phi_1)), wvar(class_size))
z_1 = np.add(np.cos(theta_1), wvar(class_size))

# class 2
theta_2 = np.random.uniform(low = np.pi/2 - np.pi/4, 
                            high = np.pi/2 + np.pi/4, 
                            size = class_size).reshape((class_size, 1))
phi_2 = np.random.uniform(low = -np.pi/4, 
                          high = np.pi/4, 
                          size = class_size).reshape((class_size, 1))
x_2 = np.add(np.multiply(np.sin(theta_2), np.sin(phi_2)), wvar(class_size))
y_2 = np.add(np.multiply(np.sin(theta_2), np.sin(phi_2)), wvar(class_size))
z_2 = np.add(np.cos(theta_2), wvar(class_size))

# class 3
theta_3 = np.random.uniform(low = np.pi/2 - np.pi/4, 
                            high = np.pi/2 + np.pi/4, 
                            size = class_size).reshape((class_size, 1))
phi_3 = np.random.uniform(low = np.pi/2 - np.pi/4, 
                          high = np.pi/2 + np.pi/4, 
                          size = class_size).reshape((class_size, 1))
x_3 = np.add(np.multiply(np.sin(theta_3), np.sin(phi_3)), wvar(class_size))
y_3 = np.add(np.multiply(np.sin(theta_3), np.sin(phi_3)), wvar(class_size))
z_3 = np.add(np.cos(theta_3), wvar(class_size))

# combine points
var_1 = np.column_stack((x_1, y_1, z_1))
var_2 = np.column_stack((x_2, y_2, z_2))
var_3 = np.column_stack((x_3, y_3, z_3))

# final dataset
dat_fin = np.vstack((var_1, var_2, var_3))
clust = np.concatenate([[0]*class_size, [1]*class_size, 
                        [2]*class_size], axis = None)
names = np.array(['One', 'Two', 'Three'])

# Plot 3-d points
ax = plt.axes(projection='3d')
# Data for three-dimensional scattered points
for i, col in enumerate(('blue', 'red', 'green')):
    ax.scatter3D(dat_fin[clust == i,0], dat_fin[clust == i,1], 
                 dat_fin[clust == i,2], label = names[i], c = col);
plt.title('Plot of Observations')
ax.set_xlabel('x')
ax.set_ylabel('y')
ax.set_zlabel('z')
ax.view_init(40, 80)
plt.show()

# 2 clusters
km_prob3_2 = kmeans(pd.DataFrame(dat_fin.copy()), 2)
names = np.array(['One', 'Two'])
ax = plt.axes(projection='3d')
for i, col in enumerate(('blue', 'red')):
    ax.scatter3D(km_prob3_2[km_prob3_2['Cluster'] == i][0], 
                 km_prob3_2[km_prob3_2['Cluster'] == i][1], 
                 km_prob3_2[km_prob3_2['Cluster'] == i][2], 
                 label=names[i], c=col)
plt.title('K-Means 2 Clusters')
ax.set_xlabel('x')
ax.set_ylabel('y')
ax.set_zlabel('z')
ax.view_init(40, 80)
plt.show()


# 3 clusters
km_prob3_3 = kmeans(pd.DataFrame(dat_fin.copy()), 3)
names = np.array(['One', 'Two', 'Three'])
ax = plt.axes(projection='3d')
for i, col in enumerate(('blue', 'red', 'green')):
    ax.scatter3D(km_prob3_3[km_prob3_3['Cluster'] == i][0], 
                 km_prob3_3[km_prob3_3['Cluster'] == i][1], 
                 km_prob3_3[km_prob3_3['Cluster'] == i][2], 
                 label=names[i], c=col)
plt.title('K-Means 3 Clusters')
ax.set_xlabel('x')
ax.set_ylabel('y')
ax.set_zlabel('z')
ax.view_init(40, 80)
plt.show()


# 4 clusters
km_prob3_4 = kmeans(pd.DataFrame(dat_fin.copy()), 4)
names = np.array(['One', 'Two', 'Three', 'Four'])
ax = plt.axes(projection='3d')
for i, col in enumerate(('blue', 'red', 'green', 'orange')):
    ax.scatter3D(km_prob3_4[km_prob3_4['Cluster'] == i][0], 
                 km_prob3_4[km_prob3_4['Cluster'] == i][1], 
                 km_prob3_4[km_prob3_4['Cluster'] == i][2], 
                 label=names[i], c=col)
plt.title('K-Means 4 Clusters')
ax.set_xlabel('x')
ax.set_ylabel('y')
ax.set_zlabel('z')
ax.view_init(40, 80)
plt.show()

# 5 clusters
km_prob3_5 = kmeans(pd.DataFrame(dat_fin.copy()), 5)
names = np.array(['One', 'Two', 'Three', 'Four', 'Five'])
ax = plt.axes(projection='3d')
for i, col in enumerate(('blue', 'red', 'green', 'orange', 'gray')):
    ax.scatter3D(km_prob3_5[km_prob3_5['Cluster'] == i][0], 
                 km_prob3_5[km_prob3_5['Cluster'] == i][1], 
                 km_prob3_5[km_prob3_5['Cluster'] == i][2], 
                 label=names[i], c=col)
plt.title('K-Means 5 Clusters')
ax.set_xlabel('x')
ax.set_ylabel('y')
ax.set_zlabel('z')
ax.view_init(40, 80)
plt.show()