import numpy as np
from sklearn.cluster import KMeans

from data_preprocessing import load_data, prepare_character_data
from feature_extraction import urls_to_embeddings_with_position, compute_mivlad

def main():
    train_csic, test_csic = load_data()
    model = prepare_character_data(train_csic)
    train_urls = train_csic['Data'].tolist()
    test_urls = test_csic['Data'].tolist()
    
    train_embeddings = urls_to_embeddings_with_position(train_urls, model)
    test_embeddings = urls_to_embeddings_with_position(test_urls, model)
    
    kmeans = KMeans(n_clusters=5, random_state=0).fit(np.vstack(train_embeddings))
    train_mivlad = compute_mivlad(train_embeddings, kmeans)
    test_mivlad = compute_mivlad(test_embeddings, kmeans)
    
    test_npy = np.column_stack((test_mivlad, test_csic['Label'].to_numpy()))
    train_npy = np.column_stack((train_mivlad, train_csic['Label'].to_numpy()))
    
    np.save('train.npy', train_npy)
    np.save('test.npy', test_npy)

if __name__ == '__main__':
    main()
