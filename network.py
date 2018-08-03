import csv
import cv2
import numpy as np
import sklearn
from random import randint

## OPEN FILES
samples = []
with open('./data/provided/driving_log.csv') as csvfile:
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
        sklearn.utils.shuffle(samples)
        for offset in range(0, num_samples, batch_size):
            batch_samples = samples[offset:offset+batch_size]
            
            images = []
            angles = []
            for batch_sample in batch_samples:
                dir = randint(0,2)
                flip = randint(0,1)
                name = './data/provided/IMG/'+batch_sample[dir].split('/')[-1]
                image = cv2.imread(name)
                if dir == 0:
                    angle = float(batch_sample[3])
                if dir == 1:
                    angle = float(batch_sample[3]) + correction
                if dir == 2:
                    angle = float(batch_sample[3]) - correction
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
from keras.layers import Conv2D, MaxPooling2D
import matplotlib.pyplot as plt

model = Sequential([
    Cropping2D(cropping=((60,20), (0,0)), input_shape=(160,320,3)),
    Lambda(lambda x: x / 255.0 - 0.5),
    Conv2D(6,(5,5),activation='relu',padding='same'),
    MaxPooling2D(),
    Conv2D(12,(5,5),activation='relu',padding='same'),
    MaxPooling2D(),
    Flatten(),
    Dense(120),
    Dense(84),
    Dense(1),
])
print(model.summary())

for layer in model.layers:
    print(layer.get_output_at(0).get_shape().as_list())


## MODEL TRAINING AND EVALUATION
model.compile(loss='mse', optimizer='adam')
history = model.fit_generator(train_generator, steps_per_epoch=len(train_samples), validation_data=validation_generator, validation_steps=len(validation_samples), epochs=5, verbose = 1)
model.save('model.h5')
print(history.history.keys())
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('model mean squared error loss')
plt.ylabel('mean squared error loss')
plt.xlabel('epoch')
plt.legend(['training set', 'validation set'], loc='upper right')
plt.show()
