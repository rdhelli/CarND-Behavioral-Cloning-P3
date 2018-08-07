import csv
import cv2
import os
import shutil
import numpy as np
import sklearn
from random import randint
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from keras.models import Sequential, load_model
from keras.layers import Flatten, Dense, Lambda, Cropping2D, Conv2D, Dropout
from keras.optimizers import Adam
from keras.callbacks import ModelCheckpoint
from keras.utils import plot_model


# CREATING INPUT DATA FOLDER FROM MULTIPLE FOLDERS
def manage_input_data(folder_names, destination_path, destination_csv):
    all_image_names = []
    all_csv = []
    for folder in folder_names:
        print('accessing', folder, 'folder...')
        # Managing images
        src_path = ('./data/' + folder + '/IMG')
        image_names = os.listdir(src_path)
        all_image_names.extend(image_names)
        for image_name in image_names:
            image_path = os.path.join(src_path, image_name)
            dest_path = os.path.join(destination_path, image_name)
            if (os.path.isfile(image_path) and not os.path.isfile(dest_path)):
                shutil.copy(image_path, destination_path)
        # Managing csv files
        csv_path = ('./data/' + folder + '/driving_log.csv')
        with open(csv_path) as csvfile:
            reader = csv.reader(csvfile)
            for line in reader:
                all_csv.append(line)
    # Preparing working directory
    print('creating summary csv file...')
    with open(destination_csv, 'w', newline='') as csvfile:
        csvfile.truncate()
        writer = csv.writer(csvfile)
        writer.writerows(all_csv)
    destination_files = os.listdir(destination_path)
    for file in destination_files:
        if file not in all_image_names:
            os.remove(os.path.join(destination_path, file))
    print('work directory ready')

# folders for separate feature driven recordings
folder_names = [
        'provided',
        # 'normal_driving_3_rounds',
        # 'reverse_driving_3_rounds',
        # 'curve_normal_2_rounds',
        # 'curve_reverse_2_rounds',
        # 'veer_to_center_normal_1_round',
        # 'veer_to_center_back_1_round',
        # 'dirt',
        # 'jungle_normal_2_rounds',
        ]
destination_path = './data/workdir/IMG'
destination_csv = './data/workdir/driving_log.csv'
manage_input_data(folder_names, destination_path, destination_csv)

# OPEN FILES
samples = []
with open('./data/workdir/driving_log.csv') as csvfile:
    reader = csv.reader(csvfile)
    for line in reader:
        samples.append(line)
train_samples, valid_samples = train_test_split(samples, test_size=0.2)

# DATA ANALYSIS
labels = [float(sample[3]) for sample in samples]
plt.figure(1)
plt.hist(labels, bins=30)
plt.title("Histogram of steering angles")
plt.show()
del labels


# INPUT DATA IN BATCHES, RANDOM AUGMENTATION
def generator(samples, batch_size=32, training=True):
    num_samples = len(samples)
    correction = 0.2
    while 1:
        sklearn.utils.shuffle(samples)
        for offset in range(0, num_samples, batch_size):
            batch_samples = samples[offset:offset+batch_size]
            images = []
            angles = []
            for batch_sample in batch_samples:
                b_imgs = []
                b_angs = []
                # Adding images from all 3 camera angles
                for pos in range(3):
                    b_imgs.append(cv2.imread('./data/workdir/IMG/'+
                            batch_sample[pos].split('\\')[-1]))
                    # Adjusting steering angle based on camera position
                    if pos == 0:
                        b_angs.append(float(batch_sample[3]))
                    if pos == 1:
                        b_angs.append(float(batch_sample[3]) + correction)
                    if pos == 2:
                        b_angs.append(float(batch_sample[3]) - correction)

                for flip_pos in range(3):
                    # Flipping image with 50% chance, adjusting steering angle
                    if training:
                        b_imgs.append(cv2.flip(b_imgs[flip_pos], 1))
                        b_angs.append(b_angs[flip_pos] * -1.0)
                if training:
                    # Horizontally translate centered images, adjusting steering angle
                    for i in range(6): # (0, 3):
                        trx = randint(-50, 50)
                        b_angs[i] += trx*0.003
                        M_tr = np.float32([[1, 0, trx], [0, 1, 0]])
                        b_imgs[i] = cv2.warpAffine(b_imgs[i], M_tr,
                              (b_imgs[i].shape[1], b_imgs[i].shape[0]))
                images.extend(b_imgs)
                angles.extend(b_angs)
            # plt.figure()
            # plt.hist(angles, bins=30)
            # plt.show()
            X_train = np.array(images)
            y_train = np.array(angles)
            yield sklearn.utils.shuffle(X_train, y_train)


# MODEL ARCHITECTURE
# Nvidia architecture
model = Sequential([
        Cropping2D(cropping=((60, 20), (0, 0)), input_shape=(160, 320, 3)),
        Lambda(lambda x: x / 127.5 - 1.0),
        Conv2D(24, (5, 5), strides=(2, 2), activation='relu'),
        Conv2D(36, (5, 5), strides=(2, 2), activation='relu'),
        Conv2D(48, (5, 5), strides=(2, 2), activation='relu'),
        Conv2D(64, (3, 3), activation='relu'),
        Conv2D(64, (3, 3), activation='relu'),
        Dropout(.5),
        Flatten(),
        Dense(100, activation='relu'),
        Dense(50, activation='relu'),
        Dense(10, activation='relu'),
        Dense(1)
])
print(model.summary())
# plot_model(model, to_file='./examples/model.png', show_shapes=True)
if (os.path.isfile('./model.h5')):
    del model
    model = load_model('./model.h5')

# MODEL TRAINING AND EVALUATION
train_generator = generator(train_samples, batch_size=64, training=True)
valid_generator = generator(valid_samples, batch_size=64, training=False)
# adding checkpoint to save model between epochs
checkpoint = ModelCheckpoint(filepath='model{epoch:02d}.h5',
                             monitor='val_loss',
                             verbose=0,
                             save_best_only=True,
                             mode='auto')

model.compile(loss='mse', optimizer=Adam(lr=0.0001))

history = model.fit_generator(generator=train_generator,
                              steps_per_epoch=5000,
                              validation_data=valid_generator,
                              validation_steps=len(valid_samples),
                              epochs=2,
                              callbacks=[checkpoint],
                              verbose=1)
model.save('model.h5')
print(history.history.keys())
plt.figure(2)
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('model mean squared error loss')
plt.ylabel('mean squared error loss')
plt.xlabel('epoch')
plt.legend(['training set', 'validation set'], loc='upper right')
plt.show()
