import numpy as np
from keras.layers import LSTM, Dense
from tensorflow import keras


if __name__ == '__main__':
    # load features and labels
    X_train = np.loadtxt("x_train1.csv")
    X_val = np.loadtxt("x_val1.csv")
    y_train = np.loadtxt("y_train1.csv")
    y_val = np.loadtxt("y_val1.csv")
    # Print the shapes
    print(X_train.shape, X_val.shape, len(y_train), len(y_val))

    # ####   our model  #####
    input_shape = (128, 1)

    model = keras.Sequential()
    model.add(LSTM(256, input_shape=input_shape))
    model.add(Dense(128, activation='relu'))
    model.add(Dense(2, activation='softmax'))

    model.summary()

    # change default learning rate
    opt = keras.optimizers.Adam(learning_rate=0.001)

    # load kept model. this is final model
    # model = keras.models.load_model("san7")
    # model.compile(optimizer=opt, loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    history = model.fit(X_train, y_train, epochs=25
                        , batch_size=72,
                        validation_data=(X_val, y_val), shuffle=False)
