#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Created on 14/04/2023 19:37
@author: hheise

This approach involves representing cells as nodes in a graph and establishing edges between cells based on their
feature distances or similarities. You can then use graph-based algorithms, such as the maximum weighted bipartite
matching algorithm, to find the optimal matching between cells. You can implement this approach using libraries such as
NetworkX or scipy.optimize in Python.

Using the Hungarian algorithm on a graph-based representation can have advantages over using it directly on a feature
space. Here are some potential advantages:

1.  Incorporation of spatial relationships: Graph-based representations can capture the spatial relationships between
    cells, which can be important for cell matching in images. For example, cells that are close to each other in one
    image are likely to be close to each other in the other image as well. By representing cells as nodes and adding
    edges between them based on spatial criteria, a graph-based representation can encode this spatial information,
    which can improve the accuracy of cell matching.

2.  Robustness to small image position differences: Graph-based representations can be more robust to small image
    position differences, as they can capture the relative spatial relationships between cells regardless of the
    absolute position of the images. This can be beneficial when dealing with images taken from slightly different
    positions or with slight tissue changes over time.

3.  Ability to handle missing or noisy data: Graph-based representations can naturally handle missing or noisy data, as
    nodes in the graph can represent cells that are visible in some images but not in others. This can be useful in
    cases where some cells may not be fluorescent or visible in certain images due to technical or biological reasons.

4.  Flexibility in incorporating additional criteria: Graph-based representations can easily incorporate additional
    criteria or features beyond the spatial relationships, such as cell size, shape, or intensity, as edges between
    nodes can be weighted based on multiple features. This can provide flexibility in designing the matching algorithm
    based on the specific characteristics of the cells or the imaging data.
"""

import numpy as np
import networkx as nx
from scipy.optimize import linear_sum_assignment

# Assume you have extracted features for each cell in two images, stored in numpy arrays
# features_img1 and features_img2, where each row represents the features for a single cell
features_img1 = np.zeros(1)
features_img2 = np.zeros(1)

# Calculate pairwise distances/similarities between the features of cells in two images
# You can choose the appropriate distance/similarity metric based on the characteristics of your features
# For example, you can use Euclidean distance for center of mass and spatial footprint, and cosine similarity for angles
distances = np.linalg.norm(features_img1 - features_img2, axis=1)  # Euclidean distance
similarities = np.dot(features_img1, features_img2.T) / (np.linalg.norm(features_img1, axis=1) * np.linalg.norm(features_img2, axis=1))[:, np.newaxis]  # Cosine similarity

# Set a threshold for matching based on the distance/similarity
threshold = 0.2  # You can adjust this threshold based on your specific dataset
threshold_feature = 0.2     # Threshold for other features. Each feature needs its own, validated threshold

# Create a graph representation of the cells using NetworkX
G = nx.Graph()

# Add nodes to the graph for cells in the first image
for i in range(features_img1.shape[0]):
    G.add_node(('img1', i))

# Add nodes to the graph for cells in the second image
for j in range(features_img2.shape[0]):
    G.add_node(('img2', j))

# Add edges between nodes in the graph based on the distance/similarity between cell features
for i in range(features_img1.shape[0]):
    for j in range(features_img2.shape[0]):
        # Check if the distance/similarity is below the threshold
        if distances[i] < threshold:
            # If yes, add an edge between the corresponding nodes in the graph
            G.add_edge(('img1', i), ('img2', j), weight=distances[i])  # You can use distances or similarities as edge weights

        # Use other feature thresholds and compare against given threshold
        if similarities[i] < threshold_feature:
            G.add_edge(('img1', i), ('img2', j), weight=similarities[i])

# Convert the graph to a matrix representation for input to the Hungarian Algorithm
cost_matrix = np.zeros((len(G.nodes), len(G.nodes)))
for u, v, data in G.edges(data=True):
    # Here, multiple edges are just summed. This only works if all edge weights in the cost matrix reflect the
    # similarity x-or dissimilarity between cells, and you may need to adapt the way they are accumulated or combined
    # depending on the specific use case and dataset characteristics.
    cost_matrix[list(G.nodes).index(u), list(G.nodes).index(v)] += data['weight']

    # # One example on how to accumulate different edges with a weighted sum. Here, each edge/feature needs its own
    # # validated weight.
    # alpha = 0.5  # Weight for distance-based edges
    # beta = 0.5  # Weight for feature-based edges
    # combined_edges = alpha * distance_edges + beta * feature_edges

# Use the Hungarian Algorithm to find the optimal matching between cells
row_indices, col_indices = linear_sum_assignment(cost_matrix)

# Initialize an array to store matched cell indices
matched_indices = []

# Iterate through the matched cell indices returned by the Hungarian Algorithm
for row_idx, col_idx in zip(row_indices, col_indices):
    # Check if the cost for the matched cell pair is below the threshold
    if cost_matrix[row_idx, col_idx] < threshold:
        # If yes, consider it as a match and store the indices of the matched cells
        img1_idx = [i for i in range(features_img1.shape[0]) if G.nodes[i] == row_idx][0]
        img2_idx = [i for i in range(features_img1.shape[0], len(G.nodes)) if G.nodes[i] == col_idx][0]
        matched_indices.append((img1_idx, img2_idx))

# Print the matched cell indices
print("Matched Cell Indices: ", matched_indices)
