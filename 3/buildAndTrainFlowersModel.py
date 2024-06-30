#!/usr/bin/env python
# coding: utf-8

# In[1]:


import tensorflow as tf


# In[2]:


import tensorflow_datasets as tfds
from tensorflow import keras
import numpy as np
from tensorflow.keras.applications import DenseNet121
from preprocessDefinition import preprocess


# In[3]:


# Load the Oxford Flowers 102 dataset
(train_ds, val_ds), info = tfds.load('oxford_flowers102',
                                     split=['train', 'validation'],
                                     as_supervised=True,
                                     with_info=True)


# In[4]:


# Preprocess datasets
train_ds = train_ds.map(preprocess, num_parallel_calls=8).batch(8).prefetch(1)
val_ds = val_ds.map(preprocess, num_parallel_calls=8).batch(8).prefetch(1)


# In[5]:


# Load the base DenseNet121 model pre-trained on ImageNet
base_model = DenseNet121(include_top=False, weights='imagenet', input_shape=(299, 299, 3))
base_model.trainable = False  # Freeze base model


# In[6]:


# Create a new model head for the Oxford Flowers 102 dataset
inputs = keras.Input(shape=(299, 299, 3))
x = base_model(inputs, training=False)
x = keras.layers.GlobalAveragePooling2D()(x)
x = keras.layers.Dropout(0.4)(x)
outputs = keras.layers.Dense(info.features['label'].num_classes, activation='softmax')(x)

# Create the model
model = keras.Model(inputs, outputs)


# In[7]:


# Optimizer
ss = 1e-3
optimizer = keras.optimizers.Adam(learning_rate=ss)

# Compile the model
model.compile(optimizer=optimizer,
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy',
                       tf.keras.metrics.SparseTopKCategoricalAccuracy(k=2, name='top2'),
                       tf.keras.metrics.SparseTopKCategoricalAccuracy(k=5, name='top5')],)

# Callbacks
checkpoint_cb = keras.callbacks.ModelCheckpoint(
    'oxFlowersMod-topFit.keras',
    save_best_only=True,
    monitor='val_accuracy',
    verbose=1
)

early_stop_cb = keras.callbacks.EarlyStopping(
    patience=5,
    restore_best_weights=True,
    monitor='val_loss',
    verbose=1
)

# Train the model
history = model.fit(train_ds, validation_data=val_ds, epochs=25,
                    callbacks=[checkpoint_cb, early_stop_cb])


# In[8]:


# Save the final model
model.save('flowersModel.keras')

