import os

os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2"  # Choose which GPUs by checking current use with nvidia-smi
import tensorflow as tf
# import tensorflow_addons as tfa
from tensorflow import keras
## Keras library also provides ResNet101V2 and ResNet50V2. Import them and use it for other experiments.
from tensorflow.keras.applications import ResNet152V2
from tensorflow.keras.applications import ResNet50V2
from tensorflow.keras.callbacks import ReduceLROnPlateau
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras import metrics
import time
import numpy as np
import pandas as pd
from tensorflow.keras.models import model_from_json
from keras import backend as K

# Check CUDA functionality, restart kernel to change GPUs
gpus = tf.config.list_physical_devices('GPU')
print("*************************************")
print(gpus)
print("*************************************")


# Define function to preprocess images as required by ResNet
def preprocess(images, labels):
    return tf.keras.applications.resnet_v2.preprocess_input(images), labels


# setup train, validation, and test folders
traindir = '/mnt/d/datasets/INBREAST/split/train'
valdir = '/mnt/d/datasets/INBREAST/split/validation'
testdir = '/mnt/d/datasets/INBREAST/split/test'
dirName = '2_classes'

buffersize = 2
# im_dim = 512
im_dim_x = 224
im_dim_y = 224
batchSizeIntInitial = 10
batchSizeInt = 32
model_name = "resnet50_with_batch_norm_extendedV3"

train = tf.keras.preprocessing.image_dataset_from_directory(
    traindir, image_size=(im_dim_x, im_dim_y), batch_size=batchSizeInt)
val = tf.keras.preprocessing.image_dataset_from_directory(
    valdir, image_size=(im_dim_x, im_dim_y), batch_size=batchSizeInt)
test = tf.keras.preprocessing.image_dataset_from_directory(
    testdir, image_size=(im_dim_x, im_dim_y), batch_size=batchSizeInt)

test_ds = test.map(preprocess)
train_ds = train.map(preprocess)
val_ds = val.map(preprocess)
train_ds = train_ds.prefetch(buffer_size=buffersize)
val_ds = val_ds.prefetch(buffer_size=buffersize)

## set up hyperparameters, such as epochs, learning rates, cutoffs.
epochs = 30
lr = 0.0004
cutoff = 0.5
start_time = time.time()
mirrored_strategy = tf.distribute.MirroredStrategy()

with mirrored_strategy.scope():  # the entire model needs to be compiled within the scope of the distribution strategy
    # cb1 = EarlyStopping(monitor='val_accuracy', patience=4)  # define early stopping callback function
    cb1 = ModelCheckpoint(f"/mnt/d/datasets/INBREAST/results/{model_name}.keras", monitor="val_accuracy", verbose=2, save_best_only=True, mode="max")
    cb2 = ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=3, min_lr=0.000001)
    # cb2 = ReduceLROnPlateau(monitor='val_accuracy', factor=0.2, patience=2,
    #                         min_lr=0.00001)  # define LR reduction callback function
    opt = keras.optimizers.Adam(learning_rate=lr)
    metr = [metrics.BinaryAccuracy(name='accuracy', threshold=cutoff), metrics.AUC(name='auc'),
            metrics.Precision(name='precision'),
            metrics.Recall(name='recall')]
    #                 tfa.metrics.F1Score(name='f1_score')]
    ptmodel = ResNet50V2(include_top=False, weights='imagenet', classes=2, input_shape=(im_dim_x, im_dim_y, 3),
                         pooling='avg')  # compile resnet152v2 with imagenet weights
    ptmodel.trainable = False  # freeze layers
    ptmodel.layers[-1].trainable = True


    #un-freeze the BatchNorm layers
    # for layer in ptmodel.layers:
    #     if "BatchNormalization" in layer.__class__.__name__:
    #         layer.trainable = False

    last_output = ptmodel.output
    x = tf.keras.layers.Flatten()(last_output)
    x = tf.keras.layers.Dense(1024, activation='relu')(x)
    x = tf.keras.layers.Dropout(0.2)(x)
    x = tf.keras.layers.Dense(256, activation='relu')(x)
    x = tf.keras.layers.Dropout(0.2)(x)
    x = tf.keras.layers.Dense(128, activation='relu')(x)
    x = tf.keras.layers.Dropout(0.2)(x)
    x = tf.keras.layers.Dense(64, activation='relu')(x)
    x = tf.keras.layers.Dense(32, activation='relu')(x)
    x = tf.keras.layers.Dense(1, activation='sigmoid')(x)
    model = tf.keras.Model(ptmodel.input, x)
    model.compile(optimizer=opt,
                  loss='BinaryCrossentropy',
                  metrics=metr)

print("---time taken : %s seconds ---" % (time.time() - start_time))
# Train model
model.summary()
history = model.fit(train_ds, validation_data=val_ds, epochs=epochs, callbacks=[cb1, cb2])
print("---time taken : %s seconds ---" % (time.time() - start_time))

# Save training subresults to csv
history_df = pd.DataFrame(history.history)
history_df.to_csv(f"/mnt/d/datasets/INBREAST/results/{model_name}_history.csv", index=False)

model = tf.keras.models.load_model(f"/mnt/d/datasets/INBREAST/results/{model_name}.keras")

# Test model
# Loading checkpoint model
testloss, testaccuracy, testauc, precision, recall = model.evaluate(test_ds)
print('Test accuracy :', testaccuracy)
print('Test AUC :', testauc)

F1 = 2 * float(precision) * float(recall) / (float(precision) + float(recall))
print('Test F1 :', F1)
print('Test precision :', precision)
print('Test recall :', recall)

test_results = {'Accuracy': testaccuracy, 'Test AUC': testauc, 'F1': F1, 'precision': precision, 'recall': recall}
results_df = pd.DataFrame(test_results)

results_df.to_csv(f"/mnt/d/datasets/EMBED/results/{model_name}_results.csv")

tf.keras.models.save_model(model, f'{model_name}.h5')
loaded_model = tf.keras.models.load_model(f'{model_name}.h5')

predicted_probs = []
loaded_probs = []
true_classes = []

for images, labels in test_ds:
    predicted_prob = model(images).numpy()
    true_class = labels.numpy()

    predicted_probs.extend(predicted_prob)
    true_classes.extend(true_class)

    loaded_prob = loaded_model(images).numpy()

    loaded_probs.extend(predicted_prob)

print(predicted_probs)
print(loaded_probs)

probs_dict = {
    'Probabilities': predicted_probs,
    'Labels': true_classes
}

probs_df = pd.DataFrame(probs_dict)
probs_df.to_csv(f"/mnt/d/datasets/EMBED/results/{model_name}_probs.csv", index=False)

