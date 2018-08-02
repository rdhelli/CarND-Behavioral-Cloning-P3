import csv
import cv2
import numpy as np
import sklearn
from random import randint

## OPEN FILES
samples = []
with open('./driving_log.csv') as csvfile:
    reader = csv.reader(csvfile)
    for line in reader:
        samples.append(line)

from sklearn.model_selection import train_test_split
train_samples, validation_samples = train_test_split(samples, test_size=0.2)

## INPUT DATA IN BATCHES, RANDOM AUGMENTATION
def generator(samples, batch_size=32):
    num_samples = len(samples)
    correction = 0.2
    while 1:
        shuffle(samples)
        for offset in range(0, num_samples, batch_size):
            batch_samples = samples[offset:offset+batch_size]
            
            images = []
            angles = []
            for batch_sample in batch_samples:
                dir = randint(0,2)
                flip = randint(0,1)
                name = './IMG/'+batch_sample[dir].split('\\')[-1]
                image = cv2.imread(name)
                if dir == 0:
                    angle = float(batch_sample[3])
                if dir == 1:
                    angle = float(batch_sample[3] + correction)
                if dir == 2:
                    angle = float(batch_sample[3] - correction)
                if flip:
                    image = cv2.flip(image,1)
                    angle *= -1.0
                images.append(image)
                angles.append(angle)
            
            X_train = np.array(images)
            y_train = np.array(images)
            yield sklearn.utils.shuffle(X_train, y_train)

train_generator = generator(train_samples, batch_size=32)
validation_generator = generator(validation_samples, batch_size=32)

## MODEL ARCHITECTURE
from keras.models import Sequential
from keras.layers import Flatten, Dense, Lambda, Cropping2D
from keras.layers import Convolution2D, MaxPooling2D
import matplotlib.pyplot as plt

model = Sequential()
model.add(Cropping2D(cropping=((50,20), (0,0)), input_shape=(160,320,3)))
model.add(Lambda(lambda x: x / 255.0 - 0.5))
model.add(Convolution2D(6,(5,5),activation='relu'))
model.add(MaxPooling2D())
model.add(Convolution2D(6,(5,5),activation='relu'))
model.add(MaxPooling2D())
model.add(Flatten())
model.add(Dense(120))
model.add(Dense(84))
model.add(Dense(1))

## MODEL TRAINING AND EVALUATION
model.compile(loss='mse', optimizer='adam')
history = model.fit_generator(train_generator,
                              samples_per_epoch=len(train_samples),
                              validation_data=validation_generator,
                              nb_val_samples=len(validation_samples),
                              nb_epoch=5)
model.save('model.h5')
print(history.history.keys())
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('model mean squared error loss')
plt.ylabel('mean squared error loss')
plt.xlabel('epoch')
plt.legend(['training set', 'validation set'], loc='upper right')
plt.show()
