# Author: Alexis Yang
import math
import string
from collections import Counter


def parse(corpus):
    table = str.maketrans(dict.fromkeys(string.punctuation + string.digits))
    corpus = [review.translate(table).split() for review in corpus.split('<br /><br />')]
    return [[word.lower().rstrip('s') for word in review if word.lower() not in stopwords]
            for review in corpus]


class NaiveBayes:

    def __init__(self):
        self.vocab = {}

    def _getVocab(self, corpus):
        self.vocab = dict.fromkeys(word for review in corpus for word in review)


class MultinomialNB(NaiveBayes):

    def _extractFeature(self, corpus):
        data = []
        self._count_sum = Counter()
        self._review_num = Counter()
        for review in corpus:
            feature_v = Counter()
            for word in review:
                if word in self.vocab:
                    feature_v[word] += 1
                    self._count_sum[word] += 1
            for word in feature_v.keys():
                self._review_num[word] += 1
            data.append(feature_v)
        return data

    def learn(self, pos_corpus, neg_corpus):
        self._getVocab(pos_corpus + neg_corpus)

        self._extractFeature(pos_corpus)
        word_num = sum(len(review) for review in pos_corpus)
        self.pos_p = {word: self._laplaceSmoothP(self._count_sum[word], word_num) for word in self.vocab.keys()}

        self._extractFeature(neg_corpus)
        word_num = sum(len(review) for review in neg_corpus)
        self.neg_p = {word: self._laplaceSmoothP(self._count_sum[word], word_num) for word in self.vocab.keys()}

    def classify(self, corpus):
        if not self.vocab:
            raise TrainingError("Attempt to predict before learning")
        data = self._extractFeature(corpus)
        predictions = []
        for x in data:
            prob_pos = sum(x[word]*math.log(self.pos_p[word]) for word in x.keys())
            prob_neg = sum(x[word]*math.log(self.neg_p[word]) for word in x.keys())
            predictions.append(prob_pos > prob_neg)
        return predictions

    def _laplaceSmoothP(self, count_sum, n, alpha=45):
        return (count_sum + alpha) / (n + alpha*len(self.vocab))


class GaussianNB(NaiveBayes):

    def _extractFeature(self, corpus):
        data = []
        self._count_sum = Counter()
        self._count_square_sum = Counter()
        self._review_num = Counter()
        for review in corpus:
            feature_v = Counter()
            for word in review:
                if word in self.vocab:
                    feature_v[word] += 1
                    self._count_sum[word] += 1
            for word in feature_v.keys():
                self._review_num[word] += 1
                self._count_square_sum[word] += feature_v[word]**2
            data.append(feature_v)
        if self.feature == 'TFIDF':
            self._count_sum = Counter()
            self._count_square_sum = Counter()
            for i in range(len(data)):
                for word in data[i].keys():
                    data[i][word] *= math.log(len(corpus) / self._review_num[word]) / len(corpus[i])
                    self._count_sum[word] += data[i][word]
                    self._count_square_sum[word] += data[i][word]**2
        return data

    def learn(self, pos_corpus, neg_corpus, feature='BoW'):
        self.feature = feature
        self._getVocab(pos_corpus + neg_corpus)

        self._extractFeature(pos_corpus)
        pos_count_square_sum = sum(self._count_square_sum.values())
        pos_count_sum = self._count_sum
        self._extractFeature(neg_corpus)
        neg_count_square_sum = sum(self._count_square_sum.values())
        neg_count_sum = self._count_sum

        word_num = sum(len(review) for review in pos_corpus) + sum(len(review) for review in neg_corpus)
        data_num = (len(pos_corpus) + len(neg_corpus)) * len(self.vocab)
        mean_shared = word_num / data_num
        self.var = (pos_count_square_sum + neg_count_square_sum) / data_num - mean_shared**2
        self.pos_mean = {word: self._estimateMAP(pos_count_sum[word], len(pos_corpus)) for word in self.vocab.keys()}
        self.neg_mean = {word: self._estimateMAP(neg_count_sum[word], len(neg_corpus)) for word in self.vocab.keys()}

    def classify(self, corpus):
        if not self.vocab:
            raise TrainingError("Attempt to predict before learning")
        data = self._extractFeature(corpus)
        predictions = []
        for x in data:
            prob_pos = sum(self._logGaussian(x[word], self.pos_mean[word], self.var) for word in x.keys())
            prob_neg = sum(self._logGaussian(x[word], self.neg_mean[word], self.var) for word in x.keys())
            predictions.append(prob_pos > prob_neg)
        return predictions

    def _estimateMAP(self, count_sum, n, mean_prior=0.5, var_prior=0.5):
        return (var_prior*count_sum + self.var*mean_prior) / (var_prior*n + self.var)

    def _logGaussian(self, x, mean, var):
        return -0.5*math.log(2*math.pi*var) - (x - mean)**2 / (2*var)


class TrainingError(Exception):
    pass


# stopwords source:
# https://github.com/stanfordnlp/CoreNLP/blob/master/data/edu/stanford/nlp/patterns/surface/stopwords.txt
stopwords = dict.fromkeys([
    "!!",
    "?!",
    "??",
    "!?",
    "`",
    "``",
    "''",
    "-lrb-",
    "-rrb-",
    "-lsb-",
    "-rsb-",
    ",",
    ".",
    ":",
    ";",
    '"',
    "'",
    "?",
    "<",
    ">",
    "{",
    "}",
    "[",
    "]",
    "+",
    "-",
    "(",
    ")",
    "&",
    "%",
    "$",
    "@",
    "!",
    "^",
    "#",
    "*",
    "..",
    "...",
    "'ll",
    "'s",
    "'m",
    "a",
    "about",
    "above",
    "after",
    "again",
    "against",
    "all",
    "am",
    "an",
    "and",
    "any",
    "are",
    "aren't",
    "as",
    "at",
    "be",
    "because",
    "been",
    "before",
    "being",
    "below",
    "between",
    "both",
    "but",
    "by",
    "can",
    "can't",
    "cannot",
    "could",
    "couldn't",
    "did",
    "didn't",
    "do",
    "does",
    "doesn't",
    "doing",
    "don't",
    "down",
    "during",
    "each",
    "few",
    "for",
    "from",
    "further",
    "had",
    "hadn't",
    "has",
    "hasn't",
    "have",
    "haven't",
    "having",
    "he",
    "he'd",
    "he'll",
    "he's",
    "her",
    "here",
    "here's",
    "hers",
    "herself",
    "him",
    "himself",
    "his",
    "how",
    "how's",
    "i",
    "i'd",
    "i'll",
    "i'm",
    "i've",
    "if",
    "in",
    "into",
    "is",
    "isn't",
    "it",
    "it's",
    "its",
    "itself",
    "let's",
    "me",
    "more",
    "most",
    "mustn't",
    "my",
    "myself",
    "no",
    "nor",
    "not",
    "of",
    "off",
    "on",
    "once",
    "only",
    "or",
    "other",
    "ought",
    "our",
    "ours ",
    "ourselves",
    "out",
    "over",
    "own",
    "same",
    "shan't",
    "she",
    "she'd",
    "she'll",
    "she's",
    "should",
    "shouldn't",
    "so",
    "some",
    "such",
    "than",
    "that",
    "that's",
    "the",
    "their",
    "theirs",
    "them",
    "themselves",
    "then",
    "there",
    "there's",
    "these",
    "they",
    "they'd",
    "they'll",
    "they're",
    "they've",
    "this",
    "those",
    "through",
    "to",
    "too",
    "under",
    "until",
    "up",
    "very",
    "was",
    "wasn't",
    "we",
    "we'd",
    "we'll",
    "we're",
    "we've",
    "were",
    "weren't",
    "what",
    "what's",
    "when",
    "when's",
    "where",
    "where's",
    "which",
    "while",
    "who",
    "who's",
    "whom",
    "why",
    "why's",
    "with",
    "won't",
    "would",
    "wouldn't",
    "you",
    "you'd",
    "you'll",
    "you're",
    "you've",
    "your",
    "yours",
    "yourself",
    "yourselves",
    "###",
    "return",
    "arent",
    "cant",
    "couldnt",
    "didnt",
    "doesnt",
    "dont",
    "hadnt",
    "hasnt",
    "havent",
    "hes",
    "heres",
    "hows",
    "im",
    "isnt",
    "its",
    "lets",
    "mustnt",
    "shant",
    "shes",
    "shouldnt",
    "thats",
    "theres",
    "theyll",
    "theyre",
    "theyve",
    "wasnt",
    "were",
    "werent",
    "whats",
    "whens",
    "wheres",
    "whos",
    "whys",
    "wont",
    "wouldnt",
    "youd",
    "youll",
    "youre",
    "youve",
])
