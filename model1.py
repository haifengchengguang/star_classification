from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten, Conv1D, MaxPooling1D
def baseline_model(subclass_amount=21):
    model = Sequential()
    # model.add(ZeroPadding1D((3,3),input_shape=(45, 1)))
    model.add(Conv1D(48, kernel_size=3, strides=4, activation='relu', input_shape=(45, 1)))
    model.add(MaxPooling1D(pool_size=3, strides=2))
    model.add(Conv1D(128, kernel_size=3, padding="same", activation='relu'))
    model.add(MaxPooling1D(pool_size=3, strides=2))
    model.add(Conv1D(192, kernel_size=3, padding="same", activation='relu'))
    model.add(Conv1D(192, kernel_size=3, padding="same", activation='relu'))
    model.add(Conv1D(128, kernel_size=3, padding="same", activation='relu'))
    model.add(MaxPooling1D(pool_size=1, strides=2))
    model.add(Flatten())
    model.add(Dropout(0.5))
    model.add(Dense(2048, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(2048, activation='relu'))
    model.add(Dense(subclass_amount, activation='softmax'))
    # plot_model(model, to_file='./model_classifier.png', show_shapes=True)
    print(model.summary())
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    return model
