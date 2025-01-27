# 1. 이미지 데이터셋 내려받기

# MNIST 데이터셋 로드
import tensorflow as tf
import tensorflow_datasets as tfds

# 데이터셋 로드
dataset, info = tfds.load('MNIST', with_info=True, as_supervised=False)
train_dataset_original = dataset['train']
test_datasets = dataset['test']

# 훈련 데이터셋과 검증 데이터셋으로 나누기
train_datasets = train_dataset_original.take(50000)
valid_datasets = train_dataset_original.skip(50000)
print(len(train_datasets))
print(len(valid_datasets))
print(len(test_datasets))

for data in train_datasets.take(5):
    print(data['image'].shape)
    print(data['label'])

# 2. 데이터 전처리

# 데이터 전처리 함수
def preprocessing(data):
    image = tf.cast(data['image'], tf.float32) / 255.0
    label = data['label']
    return image, label

# 전처리를 train_datasets에 매핑하기
BATCH_SIZE = 64
train_data = train_datasets.map(preprocessing).batch(BATCH_SIZE)
valid_data = valid_datasets.map(preprocessing).batch(BATCH_SIZE)
test_data = test_datasets.map(preprocessing).batch(BATCH_SIZE)

# train_data에서 하나 추출하여 shape 확인
for image, label in train_data.take(1):
    print(image.shape)
    print(label.shape)

# 3. 데이터 증식

from tensorflow import keras
from tensorflow.keras import layers

# 데이터 증식 단계 정의
data_augmentation = keras.Sequential(
    [
        layers.RandomFlip("horizontal"),
        layers.RandomRotation(0.1),
        layers.RandomZoom(0.2)
    ]
)

# 랜덤 증식된 훈련 이미지 출력하기
import matplotlib.pyplot as plt

plt.figure(figsize=(10, 10))
for image, _ in train_data.take(1):
  for i in range(9):
    augmented_image = data_augmentation(image)
    ax = plt.subplot(3, 3, i+1)
    plt.imshow(augmented_image[0].numpy().astype("float32"))
    plt.axis("off")

# 3. 모델 만들기

# 컨브넷 모델 만들기
from tensorflow import keras
from tensorflow.keras import layers

inputs = keras.Input(shape=(28, 28, 1))
x = data_augmentation(inputs)
x = layers.Conv2D(filters=32, kernel_size=3, activation="relu")(x)
x = layers.MaxPooling2D(pool_size=2)(x)
x = layers.Conv2D(filters=64, kernel_size=3, activation="relu")(x)
x = layers.MaxPooling2D(pool_size=2)(x)
x = layers.Conv2D(filters=128, kernel_size=3, activation="relu")(x)
x = layers.Flatten()(x)
outputs = layers.Dense(10, activation="softmax")(x)
model = keras.Model(inputs=inputs, outputs=outputs)

# 모델 요약
model.summary()

# 모델 컴파일, 옵티마이저인 adam의 학습률 변경
optimizer = tf.keras.optimizers.Adam(learning_rate=3e-5) # 1e-3 -> 3e-5
model.compile(loss="sparse_categorical_crossentropy",
              optimizer=optimizer,
              metrics="accuracy")

# 4. 모델 훈련하기, 평가

callbacks = [
    keras.callbacks.ModelCheckpoint(
        filepath="convnet_from_scratch_with_augmentation.keras",
        save_best_only=True,
        monitor="val_loss"),
    keras.callbacks.EarlyStopping(
        monitor="val_accuracy",
        patience=3)
]
history = model.fit(
    train_data,
    validation_data=(valid_data),
    epochs=10,
    callbacks=callbacks)

# 훈련 과정 정확도, 손실 그래프 그리기
import matplotlib.pyplot as plt

accuracy = history.history["accuracy"]
val_accuracy = history.history["val_accuracy"]
loss = history.history["loss"]
val_loss = history.history["val_loss"]
epochs = range(1, len(accuracy)+1)
plt.plot(epochs, accuracy, "bo", label="Training accuracy")
plt.plot(epochs, val_accuracy, "b", label="Validation accuracy")
plt.title("Training and validation accuracy")
plt.legend()
plt.figure()
plt.plot(epochs, loss, "bo", label="Training loss")
plt.plot(epochs, val_loss, "b", label="Validation loss")
plt.title("Training and validation loss")
plt.legend()
plt.figure()

# 테스트 세트에서 모델 평가하기
import time

test_model = keras.models.load_model("convnet_from_scratch_with_augmentation.keras")
start_time = time.time()
test_loss, test_acc = test_model.evaluate(test_data)
end_time = time.time()
prediction_time = end_time - start_time
print(f"테스트 정확도: {test_acc:.3f}")
print(f"prediction time : {prediction_time:.3f} s")