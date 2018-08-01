import csv
import cv2
import numpy as np

folder_names = [
#        'provided'
         'normal_driving_3_rounds',
         'reverse_driving_3_rounds',
         'veer_to_center_normal_1_round',
         'veer_to_center_back_1_round'
#        'jungle_normal_1_round',
#        'jungle_reverse_1_round',
        ]

images = []
measurements = []

for folder in folder_names:
    lines = []
    with open('./data/' + folder + '/driving_log.csv') as csvfile:
        reader = csv.reader(csvfile)
        for line in reader:
            lines.append(line)

    for line in lines:
        source_path = line[0]
        filename = source_path.split('\\')[-1]
#       filename = source_path.split('/')[-1]
        current_path_center = './data/' + folder + '/IMG/' + filename
        current_path_left = './data/' + folder + '/IMG/' + filename.replace('center','left')
        current_path_right = './data/' + folder + '/IMG/' + filename.replace('center','right')
        
        # create adjusted steering measurements for the side camera images
        steering_center = float(line[3])
        correction = 0.2 # this is a parameter to tune
        steering_left = steering_center + correction
        steering_right = steering_center - correction
        
        img_center = cv2.imread(current_path_center)
        img_left = cv2.imread(current_path_left)
        img_right = cv2.imread(current_path_right)
        images.extend((img_center, img_left, img_right))
        measurements.extend((steering_center, steering_left, steering_right))

augmented_images, augmented_measurements = [], []
for image, measurement in zip(images,measurements):
    augmented_images.append(image)
    augmented_measurements.append(measurement)
    augmented_images.append(cv2.flip(image,1))
    augmented_measurements.append(measurement*-1.0)


X_train = np.array(augmented_images)
y_train = np.array(augmented_measurements)
print("X shape:", X_train.shape)
print("y shape:", y_train.shape)


from keras.models import Sequential
from keras.layers import Flatten, Dense, Lambda, Cropping2D
from keras.layers import Convolution2D, MaxPooling2D


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

model.compile(loss='mse', optimizer='adam')
model.fit(X_train, y_train, validation_split=0.2, shuffle=True, nb_epoch=5)

model.save('model.h5')