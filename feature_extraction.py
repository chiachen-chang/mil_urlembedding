import numpy as np
import math

def position_encoding(position, d_model, math):
    angle_rads = [position / math.pow(10000, 2 * (i // 2) / d_model) for i in range(d_model)]
    angle_rads[0::2] = [math.sin(angle) for angle in angle_rads[0::2]]
    angle_rads[1::2] = [math.cos(angle) for angle in angle_rads[1::2]]
    return np.array(angle_rads)

def urls_to_embeddings_with_position(urls, model):
    d_model = model.vector_size
    embeddings = []
    for url in urls:
        char_vectors = [model.wv[char] for char in url if char in model.wv]
        char_vectors_with_position = [vec + position_encoding(i, d_model, math) for i, vec in enumerate(char_vectors)]
        embeddings.append(np.array(char_vectors_with_position))
    return embeddings

def compute_mivlad(bags, kmeans):
    miVLAD_vectors = []
    for bag in bags:
        vlad_vector = np.zeros((kmeans.n_clusters, bag.shape[1]))
        nearest_clusters = kmeans.predict(bag)
        for i, instance in enumerate(bag):
            cluster_idx = nearest_clusters[i]
            vlad_vector[cluster_idx] += (instance - kmeans.cluster_centers_[cluster_idx])
        vlad_vector = vlad_vector.flatten()
        vlad_vector = np.sign(vlad_vector) * np.sqrt(np.abs(vlad_vector))
        vlad_vector = vlad_vector / np.linalg.norm(vlad_vector)
        miVLAD_vectors.append(vlad_vector)
    return np.array(miVLAD_vectors)
