import numpy as np
from tensorflow import keras


def prep():
    ##############

    training_set = ['train-clean-100', 'train-clean-360']  # , 'dev-clean']
    # validation_set = 'dev-clean'


if __name__ == '__main__':
    x_test = np.loadtxt("x_test.csv")
    y_test = np.loadtxt("y_test.csv")

    # change default learning rate
    opt = keras.optimizers.Adam(learning_rate=0.001)

    # load kept model. this is the final model
    model = keras.models.load_model("san7")
        # Evaluate the model on the test data using `evaluate`
    print("Evaluate on test data")
    results = model.evaluate(x_test, y_test, batch_size=128)
    print("test loss, test acc:", results)

    # Generate predictions (probabilities -- the output of the last layer)
    # on new data using `predict`
    print("Generate predictions for 3 samples")
    predictions = model.predict(x_test[:3])
    print("predictions shape:", predictions.shape)