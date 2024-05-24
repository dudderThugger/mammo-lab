import tensorflow as tf
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras import metrics
from tensorflow import keras

IMAGE_RESOLUTION = (224, 224)

# setup train, validation, and test folders
traindir = '/mnt/d/datasets/INBREAST/split/train'
valdir = '/mnt/d/datasets/INBREAST/split/validation'
testdir = '/mnt/d/datasets/INBREAST/split/test'
dirName = '2_classes'

img_height = 224
img_width = 224
batch_size = 6
lr = 0.04
cutoff = 0.5
epochs = 100

train_ds = tf.keras.utils.image_dataset_from_directory(
  traindir,
  image_size=(img_height, img_width),
  batch_size=batch_size)

test_ds = tf.keras.utils.image_dataset_from_directory(
  testdir,
  image_size=(img_height, img_width),
  batch_size=batch_size)

val_ds = tf.keras.utils.image_dataset_from_directory(
  valdir,
  image_size=(img_height, img_width),
  batch_size=batch_size)

cb1 = ModelCheckpoint("/mnt/d/datasets/INBREAST/results/custom_cnn.h5", monitor="val_accuracy", verbose=2,
                      save_best_only=True, mode="max")

opt = keras.optimizers.SGD(learning_rate=lr)

def create_classification_model(IMAGE_RESOLUTION):
    input = tf.keras.layers.Input(shape=(IMAGE_RESOLUTION[0], IMAGE_RESOLUTION[1], 3))
    x = tf.keras.layers.Rescaling(1./255)(input)
    x = tf.keras.layers.Conv2D(filters=16, kernel_size=5, padding='same', activation='relu')(x)
    x = tf.keras.layers.MaxPooling2D()(x)
    x = tf.keras.layers.Conv2D(filters=32, kernel_size=3, padding='same', activation='relu')(x)
    x = tf.keras.layers.Conv2D(filters=32, kernel_size=3, padding='same', activation='relu')(x)
    x = tf.keras.layers.MaxPooling2D()(x)
    x = tf.keras.layers.Conv2D(filters=64, kernel_size=3, padding='same', activation='relu')(x)
    x = tf.keras.layers.Conv2D(filters=64, kernel_size=3, padding='same', activation='relu')(x)
    x = tf.keras.layers.AvgPool2D()(x)
    x = tf.keras.layers.Flatten()(x)
    x = tf.keras.layers.Dense(512, activation='relu')(x)
    x = tf.keras.layers.Dense(128, activation='relu')(x)
    x = tf.keras.layers.Dropout(0.5, seed=34)(x)
    x = tf.keras.layers.Dense(64, activation='relu')(x)
    x = tf.keras.layers.Dense(32, activation='relu')(x)
    output = tf.keras.layers.Dense(1, activation='sigmoid')(x)

    # x = tf.keras.layers.Conv2D(filters=16, kernel_size=3, padding='same', activation='relu')(input)
    # x = tf.keras.layers.MaxPooling2D()(x)
    # x = tf.keras.layers.Conv2D(filters=32, kernel_size=3, padding='same', activation='relu')(x)
    # x = tf.keras.layers.MaxPooling2D()(x)
    # x = tf.keras.layers.Conv2D(filters=64, kernel_size=3, padding='same', activation='relu')(x)
    # x = tf.keras.layers.MaxPooling2D()(x)
    # x = tf.keras.layers.Conv2D(filters=128, kernel_size=3, padding='same', activation='relu')(x)
    # x = tf.keras.layers.MaxPooling2D()(x)
    # x = tf.keras.layers.Conv2D(filters=32, kernel_size=3, padding='same', activation='relu')(x)
    # x = tf.keras.layers.MaxPooling2D()(x)
    # x = tf.keras.layers.Flatten()(x)
    # x = tf.keras.layers.Dense(64, activation='relu')(x)
    # x = tf.keras.layers.Dense(32, activation='relu')(x)
    # output = tf.keras.layers.Dense(1, activation='sigmoid')(x)

    return tf.keras.models.Model(inputs=input, outputs=output)


metr = [metrics.BinaryAccuracy(name='accuracy', threshold=cutoff), metrics.AUC(name='auc'),
        metrics.Precision(name='precision'),
        metrics.Recall(name='recall')]

model = create_classification_model([224, 224, 3])

model.compile(optimizer=opt,
              loss='BinaryCrossentropy',
              metrics=metr)

model.summary()

model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=epochs,
    callbacks=[cb1]
)

