import gensim.downloader as api
import pandas as pd
from gensim.models import Word2Vec


class Vectorizer:
    def __init__(self):
        # self.wv = api.load('word2vec-google-news-300')
        self.model = Word2Vec.load('word2vec.model')
        self.wv = self.model.wv

    def get_vocab(self):
        return self.wv.key_to_index

    def train_model(self):
        corpus = [['first', 'sentence'], ['second', 'sentence', 'is']]
        model = Word2Vec(corpus, min_count=1, max_vocab_size=10,sg=1) # sg = 1 skip-gram , sg = 0 CBOW
        model.save('word2vec.model')

    def get_vector(self, word):
        try:
            vec = self.wv[word]
            return vec
        except KeyError:
            print("The word 'cameroon' does not appear in this model")

    def get_similarity(self, word1, word2):
        return self.wv.similarity(word1, word2)

    def get_most_similar(self, positive, n):
        return self.wv.most_similar(positive=positive, topn=n)

    def get_doesnt_match(self, data):
        return self.wv.doesnt_match(data)


if __name__ == '__main__':
    vectorizer = Vectorizer()
    vectorizer.train_model()
    print(vectorizer.get_vocab())
