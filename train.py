from preprocess import *
import keras
from keras.models import Sequential, Model
from keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPooling2D, Input
from keras.utils import to_categorical
import wandb
from wandb.keras import WandbCallback

wandb.init()
config = wandb.config

config.max_len = 11
config.buckets = 20


# Save data to array file first
#save_data_to_array(max_len=config.max_len, n_mfcc=config.buckets)

labels = ["bed", "happy", "cat"]

# # Loading train set and test set
X_train, X_test, y_train, y_test = get_train_test()

# # Feature dimension
channels = 1
config.epochs = 50
config.batch_size = 100

num_classes = 3
print("Before ", X_train.shape)
X_train = X_train.reshape(
    X_train.shape[0], config.buckets, config.max_len, channels)
X_test = X_test.reshape(
    X_test.shape[0], config.buckets, config.max_len, channels)
print("After ", X_train.shape)

y_train_hot = to_categorical(y_train)
y_test_hot = to_categorical(y_test)

#model = Sequential()
#model.add(Flatten(input_shape=(config.buckets, config.max_len, channels)))
#model.add(Dense(num_classes, activation='softmax'))

inp = Input(shape=(config.buckets, config.max_len, channels))
conv_1 = Conv2D(512, (3,3), padding='valid', activation='relu')(inp)
max_1 = MaxPooling2D(pool_size=(2,2))(conv_1)
drop_1 = Dropout(0.30)(max_1)

conv_2 = Conv2D(175, (3,3), padding='valid', activation='relu')(drop_1)
#max_2 = MaxPooling2D(pool_size=(2,2))(conv_2)
drop_2 = Dropout(0.30)(conv_2)

conv_3 = Conv2D(96, (3,3), padding='same', activation='relu')(drop_2)
drop_3 = Dropout(0.30)(conv_3)

flat_1 = Flatten()(drop_3)
drop_6 = Dropout(0.3)(flat_1)
dense_1 = Dense(200, activation='relu')(drop_6)
drop_4 = Dropout(0.30)(dense_1)
dense_2 = Dense(100, activation="relu")(drop_4)
drop_5 = Dropout(0.25)(dense_2)
dense_3 = Dense(num_classes, activation='softmax')(drop_5)
model = Model(inp, dense_3)

model.compile(loss="categorical_crossentropy",
              optimizer="adam",
              metrics=['accuracy'])
config.total_params = model.count_params()


model.fit(X_train, y_train_hot, batch_size=config.batch_size, epochs=config.epochs, validation_data=(X_test, y_test_hot), callbacks=[WandbCallback(data_type="image", labels=labels)])

