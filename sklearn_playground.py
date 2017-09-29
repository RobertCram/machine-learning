import utilities as utl
import numpy as np
import data_preprocessing as dpp
from sklearn.neural_network import MLPClassifier

def get_results(classifier, X, T):
    score = classifier.score(X, T)
    Pc = np.argmax(classifier.predict(X), axis=1)
    return score, Pc

def train():
    utl.print_title('Getting data...')
    X, Tc, X_test, Tc_test = dpp.getdata_arnold()
    #X, Tc, X_test, Tc_test = dpp.getdata_mnist()

    utl.print_title('Preparing data...')
    X, X_test = dpp.scale_data(X, X_test)
    T = dpp.one_hot_encode(Tc)
    T_test = dpp.one_hot_encode(Tc_test)

    utl.print_title('Sanity checks...')
    print('Shape X:', X.shape)
    print('Shape Tc:', Tc.shape)
    print('Shape T:', T.shape)
    print('Shape X_test:', X_test.shape)
    print('Shape Tc_test:', Tc_test.shape)
    print('Shape T_test:', T_test.shape)

    utl.print_title('Training the network...')
    classifier = MLPClassifier(solver='adam', learning_rate_init=1e-3, hidden_layer_sizes=(100), verbose=True, max_iter=200)
    classifier.fit(X, T)

    train_score, Pc = get_results(classifier, X, T)
    test_score, Pc_test = get_results(classifier, X_test, T_test)

    utl.print_title('Results:')
    print('Classification counts train (target):     ',  np.bincount(Tc.reshape(-1)))
    print('Classification counts train (prediction): ',  np.bincount(Pc))

    print('\nClassification counts test (target):     ',  np.bincount(Tc_test.reshape(-1)))
    print('Classification counts test (prediction): ',  np.bincount(Pc_test))

    print('\nTrain score: ', train_score)
    print('Test score:  ', test_score)

def main():
    train()

if __name__ == '__main__':
    main()
