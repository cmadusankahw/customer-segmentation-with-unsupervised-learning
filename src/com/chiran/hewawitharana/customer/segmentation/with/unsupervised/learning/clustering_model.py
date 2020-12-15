from sklearn.preprocessing import StandardScaler, normalize
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
import numpy as np
import pandas as pd

# The elbow method is a heuristic method of interpretation and validation of consistency within cluster analysis designed to help find the appropriate number of clusters in a dataset.
# If the line chart looks like an arm, then the "elbow" on the arm is the value of k that is the best.
# Source:
# https://en.wikipedia.org/wiki/Elbow_method_(clustering)
# https://www.geeksforgeeks.org/elbow-method-for-optimal-value-of-k-in-kmeans/

