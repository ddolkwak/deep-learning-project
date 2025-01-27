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

# make VGG16 convnet layers : 마지막에서 네번째 층까지 모두 동결

conv_base = keras.applications.vgg16.VGG16(
    weights='imagenet',
    include_top=False)

conv_base.trainable = True

for layer in conv_base.layers[:-4]:
  layer.trainable = False

len(conv_base.trainable_weights)  # 합성곱 기반 층을 동결한 후의 훈련 가능한 가중치 개수

conv_base.summary()

# 데이터 증식 단계와 밀집 분류기를 합성곱 기반 층에 추가하기

data_augmentation = keras.Sequential(
    [
        layers.RandomFlip('horizontal'),
        layers.RandomRotation(0.1),
        layers.RandomZoom(0.2)
    ]
)

inputs = keras.Input(shape=(224, 224, 3))
x = data_augmentation(inputs)  # 데이터 증식 적용
x = keras.applications.vgg16.preprocess_input(x)  # 입력 값의 스케일 조정
x = conv_base(x)
x = layers.Flatten()(x)
x = layers.Dense(50)(x)
outputs = layers.Dense(10, activation='softmax')(x)
model = keras.Model(inputs, outputs)

# Fine tuning
model.compile(loss='sparse_categorical_crossentropy',
              optimizer=keras.optimizers.Adam(learning_rate=1e-5),
              metrics=['accuracy'])

checkpoint = [
    keras.callbacks.ModelCheckpoint(
        filepath='mnist_augmentation',
        save_best_only=True,
        monitor='val_loss'
    )
]
early_stopping = keras.callbacks.EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True)

history = model.fit(
    train_set,
    epochs=5,
    steps_per_epoch = len(train_set_raw) / batch_size,
    validation_data=valid_set,
    callbacks=[checkpoint, early_stopping]
)

model.summary()

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

import time

test_model = keras.models.load_model(
    'mnist_augmentation'
)

start_time = time.time()
test_loss, test_acc = test_model.evaluate(test_set)
end_time = time.time()
prediction_time = end_time - start_time
print(f'prediction time: {prediction_time}')