# Author: Alexis Yang
import sys
import time
from nlputils import *


def getAccuracy(correct_count):
    return correct_count / (len(test_pos) + len(test_neg))


if __name__ == "__main__":
    if len(sys.argv) != 5:
        raise Exception("Two training files and two test files required")

    print('Estimated runtime: 10 seconds')
    print('\nParsing texts...')
    with open(sys.argv[1], 'r') as f:
        training_pos = parse(f.read());
    with open(sys.argv[2], 'r') as f:
        training_neg = parse(f.read());
    with open(sys.argv[3], 'r') as f:
        test_pos = parse(f.read());
    with open(sys.argv[4], 'r') as f:
        test_neg = parse(f.read());
    print('Done.')

    AI = MultinomialNB()
    print('\nTraining Multinomial Naive Bayes...')
    start = time.time()
    AI.learn(training_pos, training_neg)
    print('Training time: {} seconds'.format(time.time() - start))
    print('\nClassifying...')
    start = time.time()
    classifications_pos = AI.classify(test_pos)
    classifications_neg = AI.classify(test_neg)
    print('Classification time: {} seconds'.format(time.time() - start))
    correct_count_pos = sum(classifications_pos)
    correct_count_neg = sum(not c for c in classifications_neg)
    print(sys.argv[3] + ' classifications:')
    print('Total review count: {}\nPositive review count: {}'.format(len(test_pos), correct_count_pos))
    print(sys.argv[4] + ' classifications:')
    print('Total review count: {}\nNegative review count: {}'.format(len(test_neg), correct_count_neg))
    print("Multinomial Naive Bayes w/ BoW accuracy:", getAccuracy(correct_count_pos + correct_count_neg))

    AI = GaussianNB()
    print('\nTraining Gaussian Naive Bayes...')
    start = time.time()
    AI.learn(training_pos, training_neg, feature='BoW')
    print('Training time: {} seconds'.format(time.time() - start))
    print('\nClassifying...')
    start = time.time()
    classifications_pos = AI.classify(test_pos)
    classifications_neg = AI.classify(test_neg)
    print('Classification time: {} seconds'.format(time.time() - start))
    correct_count_pos = sum(classifications_pos)
    correct_count_neg = sum(not c for c in classifications_neg)
    print(sys.argv[3] + ' classifications:')
    print('Total review count: {}\nPositive review count: {}'.format(len(test_pos), correct_count_pos))
    print(sys.argv[4] + ' classifications:')
    print('Total review count: {}\nNegative review count: {}'.format(len(test_neg), correct_count_neg))
    print("Gaussian Naive Bayes w/ BoW accuracy:", getAccuracy(correct_count_pos + correct_count_neg))

    print('\nRetraining Gaussian Naive Bayes w/ TFIDF...')
    start = time.time()
    AI.learn(training_pos, training_neg, feature='TFIDF')
    print('Training time: {} seconds'.format(time.time() - start))
    print('\nClassifying...')
    start = time.time()
    classifications_pos = AI.classify(test_pos)
    classifications_neg = AI.classify(test_neg)
    print('Classification time: {} seconds'.format(time.time() - start))
    correct_count_pos = sum(classifications_pos)
    correct_count_neg = sum(not c for c in classifications_neg)
    print(sys.argv[3] + ' classifications:')
    print('Total review count: {}\nPositive review count: {}'.format(len(test_pos), correct_count_pos))
    print(sys.argv[4] + ' classifications:')
    print('Total review count: {}\nNegative review count: {}'.format(len(test_neg), correct_count_neg))
    print("Gaussian Naive Bayes w/ TFIDF accuracy:", getAccuracy(correct_count_pos + correct_count_neg))

