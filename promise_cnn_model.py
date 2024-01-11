from keras.models import Sequential
from keras.layers import Dense, Conv2D, Flatten, Dropout
import numpy as np
import pandas as pd
def cnn_with_dropout(original_data, original_X, original_Y, combined_training_data, x_train1, x_train2, x_train, x_test,
                     x_val, y_train1, y_train2, y_train, y_test, y_val):
    # Assuming x_train_matrix and x_val_matrix are your input data
    x_train_matrix = x_train.values
    x_val_matrix = x_val.values
    y_train_matrix = y_train.values
    y_val_matrix = y_val.values

    # Assuming y_train has a column named 'defects'
    y_train_series = y_train['defects']
    y_val_series = y_val['defects']

    # Add 5 columns of zeros to the end
    x_train_matrix = np.hstack((x_train_matrix, np.zeros((x_train_matrix.shape[0], 5))))
    x_val_matrix = np.hstack((x_val_matrix, np.zeros((x_val_matrix.shape[0], 5))))

    # Reshape to 5x5 matrix
    img_rows, img_cols = 5, 5
    x_train1 = x_train_matrix.reshape(x_train_matrix.shape[0], img_rows, img_cols, 1)
    x_val1 = x_val_matrix.reshape(x_val_matrix.shape[0], img_rows, img_cols, 1)
    input_shape = (img_rows, img_cols, 1)

    # Create a Sequential model
    model = Sequential()

    # Add convolutional layers with ReLU activation
    model.add(Conv2D(64, kernel_size=(1, 1), activation='relu', input_shape=input_shape))
    model.add(Conv2D(32, kernel_size=(3, 3), activation='relu'))

    # Dropout to mitigate overfitting
    model.add(Dropout(0.5))

    model.add(Conv2D(16, kernel_size=(4, 4), activation='relu'))

    # Dropout to mitigate overfitting
    model.add(Dropout(0.5))

    # Flatten the output of the last convolution layer
    model.add(Flatten())

    # Add dense layers with ReLU activation and Dropout
    model.add(Dense(8, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(1, activation='sigmoid'))

    # Compile the model using binary_crossentropy as the loss function and Adam optimizer
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

    # Train the model
    model.fit(x_train1, y_train_matrix, epochs=40)

    # Make predictions on the validation set
    y_pred = model.predict(x_val1) > 0.5
    y_pred_df = pd.DataFrame(y_pred)


    return model

# NN_clf = NN()
# rf_clf = random_forest()
# svm_clf = svm()
# cnn_clf = cnn()



