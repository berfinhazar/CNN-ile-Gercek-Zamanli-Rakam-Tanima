import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.datasets import mnist
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import matplotlib
matplotlib.use("TkAgg")
import matplotlib.pyplot as plt

(x_train,y_train), (x_test,y_test) = mnist.load_data()
x_train=255-x_train
x_test=255-x_test

#görselleştirmek için
plt.figure(figsize=(9,3))
for i in range(3):
    plt.subplot(1,3,i+1)
    plt.imshow(x_train[i],cmap="gray")
    plt.title(f"Label: {y_train[i]}")
    plt.axis("off")
plt.tight_layout()
plt.show()

#normalizasyon
x_train=x_train.reshape(-1,28,28,1).astype("float32")/255.0
x_test=x_test.reshape(-1,28,28,1).astype("float32")/255.0

#data eugmentation
datagen=ImageDataGenerator(
    rotation_range=10,
    width_shift_range=0.1,
    height_shift_range=0.1,
    zoom_range=0.1,
)
#model oluşturma
model=models.Sequential([

    layers.Conv2D(filters=32,kernel_size=(3,3),activation="relu",input_shape=(28,28,1)),
    layers.MaxPooling2D(pool_size=(2,2)),

    layers.Conv2D(filters=64,kernel_size=(3,3),activation="relu"),
    layers.MaxPooling2D(pool_size=(2,2)),

    layers.Flatten(),
    layers.Dense(64,activation="relu"),
    layers.Dense(10,activation="softmax")
])
print(model.summary())

#modeli derle
model.compile(optimizer="adam",loss="sparse_categorical_crossentropy",metrics=["accuracy"])

#model eğitimi ve kaydetme
model.fit(datagen.flow(x_train,y_train,batch_size=64),
          epochs=10,
          validation_data=(x_test,y_test))
#data_model_version.h5
model.save("mnist_cnn_v1.h5")
print("model başarıyla kaydedildi")