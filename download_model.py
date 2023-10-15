import nltk
import gensim.downloader

if __name__ == '__main__':
    nltk.download('punkt')
    nltk.download('brown')
    nltk.download('stopwords')
    nltk.download('averaged_perceptron_tagger')
    model = gensim.downloader.load('word2vec-google-news-300')
    model.save('data/word2vec.wordvectors')