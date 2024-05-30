import pandas as pd

def load_data():
    train_csic = pd.read_csv('sample_train.csv')
    test_csic = pd.read_csv('sample_test.csv')
    return train_csic, test_csic

def prepare_character_data(train_csic):
    from gensim.models import Word2Vec
    character_df = train_csic.copy()
    character_df['Data'] = character_df['Data'] + character_df['URI']
    character_df = character_df['Data'].tolist()
    data = [list(url) for url in character_df]
    model = Word2Vec(sentences=data, vector_size=100, window=3, min_count=2, workers=4,sg=1)
    return model
