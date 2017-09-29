import utilities as utl
import numpy as np
import data_preprocessing as dpp
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout
from keras.optimizers import Adam

def build_model(input_nodes, hidden_nodes, output_nodes):
    adam = Adam(lr=1e-3)
    model = Sequential()
    model.add(Dense(hidden_nodes, kernel_initializer='he_normal', input_dim=input_nodes, activation='relu'))
    model.add(Dense(output_nodes, kernel_initializer='he_normal', activation='softmax'))
    model.compile(loss='categorical_crossentropy', optimizer=adam, metrics=['accuracy'])
    return model

def get_results(classifier, X, T):
    score = classifier.evaluate(X, T, verbose=0)[1]
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

    utl.print_title('Training the network...')
    classifier = build_model(X.shape[1], 100, T.shape[1])
    classifier.fit(X, T, verbose=2, epochs=50, batch_size=200)

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





