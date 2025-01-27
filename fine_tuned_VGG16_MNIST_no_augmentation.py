import tensorflow as tf
import tensorflow_datasets as tfds

dataset, info = tfds.load("mnist", as_supervised=True, with_info=True)

class_names = info.features['label'].names
class_names

n_classes = info.features['label'].num_classes
n_classes

train_set_size = info.splits['train'].num_examples
test_set_size = info.splits['test'].num_examples

# Traninig set / Validation set separation 
train_set_size, test_set_size

test_set_raw, valid_set_raw, train_set_raw = tfds.load('mnist', as_supervised=True, split=['test[0%:]', 'train[:16.666%]', 'train[16.666%:]'])
len(train_set_raw), len(valid_set_raw), len(valid_set_raw)

# data preprocessing
import keras

def preprocess(image, label):
  resized_image = tf.image.resize(image, [224, 224])
  image = tf.image.grayscale_to_rgb(resized_image)
  final_image = keras.applications.vgg16.preprocess_input(image)
  return final_image, label


batch_size = 50

train_set = train_set_raw.shuffle(1000).repeat()
train_set = train_set_raw.map(preprocess).batch(batch_size).prefetch(1)

valid_set = valid_set_raw.map(preprocess).batch(batch_size).prefetch(1)

test_set = test_set_raw.map(preprocess).batch(batch_size).prefetch(1)

# make VGG16 convnet layers

conv_base = keras.applications.vgg16.VGG16(
    weights='imagenet',
    include_top=False)

conv_base.trainable = True

for layer in conv_base.layers[:-4]:
  layer.trainable = False

conv_base.summary()

# 추출

import numpy as np


def get_features_and_labels(dataset):
  all_features = []
  all_labels = []

  for images, labels in dataset:
    preprocessed_images = keras.applications.vgg16.preprocess_input(images)
    features = conv_base.predict(preprocessed_images)
    all_features.append(features)
    all_labels.append(labels)

  return np.concatenate(all_features), np.concatenate(all_labels)  # [[1,0], [2,1]] >> [1, 0, 1, 2]

val_features, val_labels = get_features_and_labels(valid_set)

train_features, train_labels = get_features_and_labels(train_set)

train_features.shape

# 밀집 연결 분류기 정의 및 훈련

from keras import layers

inputs = keras.Input(shape=(7, 7, 512))
x = layers.Flatten()(inputs)  #  Dense 층에 특성을 주입하기 전 Flatten 층을 사용
x = layers.Dense(50)(x)
x = layers.Dropout(0.5)(x)
outputs = layers.Dense(10, activation='softmax')(x)

model = keras.Model(inputs, outputs)

model.compile(loss='sparse_categorical_crossentropy',
              optimizer=keras.optimizers.Adam(learning_rate=1e-5),
              metrics=['accuracy'])

checkpoint = [
    keras.callbacks.ModelCheckpoint(
        filepath='mnist_no_augmentation',
        save_best_only=True,
        monitor='val_loss'
    )
]
early_stopping = keras.callbacks.EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True)

history = model.fit(
    train_features, train_labels,
    epochs=5,
    steps_per_epoch = len(train_set_raw) / batch_size,
    validation_data=(val_features, val_labels),
    callbacks=[checkpoint, early_stopping]
)

# 결과를 그래프로 나타내기
import matplotlib.pyplot as plt

acc = history.history['accuracy']
val_acc = history.history['val_accuracy']
loss = history.history['loss']
val_loss = history.history['val_loss']

epochs = range(1, len(acc)+1)

plt.plot(epochs, acc, 'bo', label='Traning accuracy')
plt.plot(epochs, val_acc, 'b', label='Validation accuracy')
plt.title('Training and validation accuracy')
plt.legend()

plt.figure()
plt.plot(epochs, loss, 'bo', label='Training loss')
plt.plot(epochs, val_loss, 'b', label='Validation loss')
plt.title('Training and validation loss')

plt.show()

# 테스트 세트에서 모델 평가

test_features, test_labels = get_features_and_labels(test_set)

import time

test_model = keras.models.load_model(
    'mnist_no_augmentation'
)

start_time = time.time()
test_loss, test_acc = test_model.evaluate(test_features, test_labels)
end_time = time.time()
prediction_time = end_time - start_time
print(f'prediction time: {prediction_time}')