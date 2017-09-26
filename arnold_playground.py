import numpy as np
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.neural_network import MLPClassifier

# Naming Conventions
# X = input
# Y = output
# Tc = target (classifications)
# T = target (one hot encodings)
# P = prediction (one hot encoding probabilities)
# Pc = prediction (classifications)

def getdata_arnold():
    TEST_SAMPLES = 600
    Raw = np.loadtxt('data/arnold_ka-data.csv', delimiter=',')
    np.random.shuffle(Raw)
    X = Raw[:-TEST_SAMPLES,0:-1]
    Tc = Raw[:-TEST_SAMPLES,-1].reshape(-1,1).astype(np.int32)
    X_test = Raw[-TEST_SAMPLES:,0:-1]
    Tc_test = Raw[-TEST_SAMPLES:,-1].reshape(-1,1).astype(np.int32)
    return X, Tc, X_test, Tc_test

def getdata_mnist():
    Raw = np.loadtxt('data/mnist_train.csv'.format(type), delimiter=',')
    X = Raw[:,1:]
    Tc = Raw[:,0].reshape(-1,1).astype(np.int32)
    Raw_test = np.loadtxt('data/mnist_test.csv'.format(type), delimiter=',')
    X_test = Raw_test[:,1:]
    Tc_test = Raw_test[:,0].reshape(-1,1).astype(np.int32)
    return X, Tc, X_test, Tc_test

def one_hot_encode(Yc):
    ohe = OneHotEncoder(sparse=False)
    return ohe.fit_transform(Yc)

def scale_data(X, X_test):
    scaler = StandardScaler()
    scaler.fit(X)
    return scaler.transform(X), scaler.transform(X_test)

def get_results(classifier, X, T):
    score = classifier.score(X, T)
    Pc = np.argmax(classifier.predict(X), axis=1)
    return score, Pc

def print_title(boldstring):
    print('\n\033[01m{}\033[0m'.format(boldstring))

def train():
    print_title('Getting data...')
    X, Tc, X_test, Tc_test = getdata_arnold()
    #X, Tc, X_test, Tc_test = getdata_mnist()

    print_title('Preparing data...')
    X, X_test = scale_data(X, X_test)
    T = one_hot_encode(Tc)
    T_test = one_hot_encode(Tc_test)

    print_title('Sanity checks...')
    print('Shape X:', X.shape)
    print('Shape Tc:', Tc.shape)
    print('Shape T:', T.shape)
    print('Shape X_test:', X_test.shape)
    print('Shape Tc_test:', Tc_test.shape)
    print('Shape T_test:', T_test.shape)

    print_title('Training the network...')
    classifier = MLPClassifier(solver='adam', hidden_layer_sizes=(100), alpha=1e-6, verbose=True, max_iter=200)
    classifier.fit(X, T)

    train_score, Pc = get_results(classifier, X, T)
    test_score, Pc_test = get_results(classifier, X_test, T_test)

    print_title('Results:')
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
