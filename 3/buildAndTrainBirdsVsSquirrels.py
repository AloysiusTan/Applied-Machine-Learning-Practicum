#!/usr/bin/env python
# coding: utf-8

# In[1]:


import tensorflow as tf


# In[2]:


import tensorflow_datasets as tfds
from tensorflow import keras
import numpy as np
from tensorflow.keras.applications import Xception
from preprocessDefinition import preprocess


# In[3]:


def load_dataset(file_path, batch_size=16, preprocess_fn=preprocess):
    raw_dataset = tf.data.TFRecordDataset(file_path)
    dataset = raw_dataset.map(parse_examples, num_parallel_calls=8)
    dataset = dataset.map(preprocess_fn, num_parallel_calls=8)
    return dataset.batch(batch_size).prefetch(1)

def parse_examples(serialized_examples):
    examples = tf.io.parse_example(serialized_examples, feature_description)
    targets = examples['label']
    images = tf.cast(tf.io.decode_jpeg(examples['image'], channels=3), tf.float32)
    images = tf.image.resize_with_pad(images, 299, 299)
    return images, targets


# In[4]:


feature_description = {
    'image': tf.io.FixedLenFeature([], tf.string),
    'label': tf.io.FixedLenFeature([], tf.int64)
}
    
# Load both training and validation datasets
train_dataset = load_dataset('birds-vs-squirrels-train.tfrecords', batch_size=8)
val_dataset = load_dataset('birds-vs-squirrels-validation.tfrecords', batch_size=8)


# In[5]:


base_model = keras.applications.Xception(include_top=False, weights='imagenet', input_shape=(299, 299, 3))
base_model.trainable = False  

# Add new classifier layers
inputs = keras.Input(shape=(299, 299, 3))
x = base_model(inputs, training=False)
x = keras.layers.GlobalAveragePooling2D()(x)
x = keras.layers.Dropout(0.5)(x)
outputs = keras.layers.Dense(3, activation='softmax')(x)  # Three categories

model = keras.Model(inputs, outputs)


# In[6]:


lr = 1e-3
optimizer = keras.optimizers.Adam(learning_rate=lr)
model.compile(optimizer=optimizer,
                loss='sparse_categorical_crossentropy',
                metrics=['accuracy'])

# Callbacks
checkpoint_cb = keras.callbacks.ModelCheckpoint(
    f'birdsVsSquirrelsMod-topFit.keras',
    save_best_only=True,
    monitor='val_accuracy',
    verbose=1
)
early_stop_cb = keras.callbacks.EarlyStopping(
    patience=10,
    restore_best_weights=True,
    monitor='val_loss',  
    verbose=1
)

# Train the model
history = model.fit(train_dataset, validation_data=val_dataset, epochs=25, callbacks=[checkpoint_cb, early_stop_cb])


# In[7]:


model.save(f'birdsVsSquirrelsModel.keras')

