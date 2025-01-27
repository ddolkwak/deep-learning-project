# 가위바위보 데이터셋 로드
import tensorflow as tf
import tensorflow_datasets as tfds

dataset, info = tfds.load('rock_paper_scissors', with_info=True, as_supervised=False)
train_dataset_original = dataset['train']
test_datasets = dataset['test']

# 훈련 및 검증 데이터셋 split 비율 설정
train_size = 0.85
train_count = int(info.splits['train'].num_examples * train_size)
valid_count = info.splits['train'].num_examples - train_count

# 훈련 데이터셋과 검증 데이터셋으로 나누기
train_datasets = train_dataset_original.take(train_count)
valid_datasets = train_dataset_original.skip(train_count)

# 각 데이터 셋 크기 확인
print(len(train_datasets))
print(len(valid_datasets))
print(len(test_datasets))

# 훈련 데이터의 이미지 shpae와 label 확인
for data in train_datasets.take(5):
  print(data['image'].shape)
  print(data['label'])

# 데이터 전처리 함수 생성
def preprocessing(data):
  image = tf.cast(data['image'], tf.float32) / 255.0
  label = data['label']
  return image, label

# 배치 사이즈 지정, 전처리 함수를 datasets에 매핑, 셔플 적용
BATCH_SIZE=32
train_data = train_datasets.map(preprocessing).shuffle(1000).batch(BATCH_SIZE)
valid_data = valid_datasets.map(preprocessing).batch(BATCH_SIZE)
test_data = test_datasets.map(preprocessing).batch(BATCH_SIZE)

# 데이터 증식 단계 정의
from tensorflow import keras
from tensorflow.keras import layers

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

# 모델 만들기
from tensorflow import keras
from tensorflow.keras import layers

inputs = keras.Input(shape=(300, 300, 3))
x = data_augmentation(inputs)
x = layers.Conv2D(filters=32, kernel_size=3, activation="relu")(x)
x = layers.MaxPooling2D(pool_size=2)(x)
x = layers.Conv2D(filters=64, kernel_size=3, activation="relu")(x)
x = layers.MaxPooling2D(pool_size=2)(x)
x = layers.Conv2D(filters=128, kernel_size=3, activation="relu")(x)
x = layers.MaxPooling2D(pool_size=2)(x)
x = layers.Conv2D(filters=256, kernel_size=3, activation="relu")(x)
x = layers.MaxPooling2D(pool_size=2)(x)
x = layers.Conv2D(filters=256, kernel_size=3, activation="relu")(x)
x = layers.Flatten()(x)
x = layers.Dropout(0.5)(x)
outputs = layers.Dense(3, activation="softmax")(x)
model = keras.Model(inputs=inputs, outputs=outputs)

#모델 요약
model.summary()

# 모델 컴파일. 옵티마이저인 adam의 학습률 변경
optimizer = tf.keras.optimizers.Adam(learning_rate=5e-5) # Adam의 기본 learning rate 1e-3 -> 5e-5
model.compile(loss="sparse_categorical_crossentropy",
              optimizer=optimizer,
              metrics="accuracy")

# 콜백함수 및 훈련
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
    epochs=30,
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
predict_time = end_time - start_time
print(f"테스트 정확도: {test_acc:.3f}")
print(f"predict time : {predict_time:.3f} s")