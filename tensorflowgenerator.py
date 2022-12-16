import pandas as pd
import numpy as np
from random import random
import plotext as plt
import tensorflow as tf


newlist1=[]
newlisty=[]

random1last=1

i=0
while i < (200+1):

  random1=round(6*random()+1)
  y=random1

  if i != 0:
      newlisty.append(y)
      newlist1.append(random1last)
  random1last=random1

  i=i+1

print(newlist1)

i=0
newarray=np.zeros([200, 200, 1])
newarrayy=np.zeros([200, 1])

while i < 200:
    newarrayy[i][0]=newlisty[i]-1
    j=0
    while j <= i:
        newarray[i][j][0]=newlist1[j]
        j=j+1
    i=i+1

print(newarray)
print(newarrayy)
print(np.shape(newarray))
print(np.shape(newarrayy))


newdataset = tf.data.Dataset.from_tensor_slices((newarray, newarrayy))
#newdataset = tf.data.Dataset.from_tensor_slices((arrayinone, arrayinoney))
newdataset = newdataset.batch(batch_size=40)

model = tf.keras.Sequential([
    #tf.keras.layers.Input(shape=(200,)),
    tf.keras.layers.Embedding(8,24,input_length=200),
    #tf.keras.layers.Input(shape=(None,2)),
    #tf.keras.layers.Input(shape=(10,2,)),
    #tf.keras.layers.Flatten(input_shape=[1,]),
    #tf.keras.layers.AlphaDropout(rate=0.2),
    #tf.keras.layers.BatchNormalization(),
    #tf.keras.layers.GRU(10, return_sequences=True),
    #tf.keras.layers.GRU(10, return_sequences=True),
    #tf.keras.layers.SimpleRNN(10, return_sequences=True),
    #tf.keras.layers.SimpleRNN(10, return_sequences=True),
    tf.keras.layers.Conv1D(20,6, dilation_rate=2),
    tf.keras.layers.BatchNormalization(),
    tf.keras.layers.Conv1D(20,3, dilation_rate=2),
    tf.keras.layers.BatchNormalization(),
    tf.keras.layers.Conv1D(20,2, dilation_rate=2),
    tf.keras.layers.BatchNormalization(),
    tf.keras.layers.Conv1D(20,1, dilation_rate=2),
    tf.keras.layers.BatchNormalization(),
    tf.keras.layers.LSTM(30, return_sequences=True),
    tf.keras.layers.BatchNormalization(),
    tf.keras.layers.LSTM(10, return_sequences=False),
    #tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(20, activation="relu"),
    #tf.keras.layers.AlphaDropout(rate=0.2),
    #tf.keras.layers.BatchNormalization(),
    tf.keras.layers.Dense(10, activation="relu"),
    #tf.keras.layers.AlphaDropout(rate=0.2),
    #tf.keras.layers.BatchNormalization(),
    #tf.keras.layers.Dense(3, activation="relu")
    #tf.keras.layers.TimeDistributed(tf.keras.layers.Dense(3, activation="relu"))
    tf.keras.layers.Dense(7, activation="softmax")
    ])

model.compile(
  optimizer=tf.keras.optimizers.Adam(),
  loss=tf.losses.SparseCategoricalCrossentropy(from_logits=False),
  #loss=tf.losses.SparseCategoricalCrossentropy(from_logits=False),
  #loss="binary_crossentropy",
  #metrics=[tf.metrics.MeanAbsolutePercentageError()]
  metrics=['accuracy'])

model.summary()

early_stopping_callback = tf.keras.callbacks.EarlyStopping(patience=10)
best_checkpoint_callback = tf.keras.callbacks.ModelCheckpoint("sequence_model.h5", save_best_only=True)
learningratecallbackchange=tf.keras.callbacks.LearningRateScheduler(lambda epoch: 0.0015 * 0.9 ** epoch)


fittingdiagram=model.fit(
  newdataset,
  #validation_data=val_ds,
  epochs=1000,
  #callbacks=[best_checkpoint_callback, early_stopping_callback, learningratecallbackchange]
  #callbacks=[best_checkpoint_callback, early_stopping_callback]
  )

i=0
predictionarray=np.zeros([1,200,1])
predictionarray[0][0][0]=2
firstprediction=(np.argmax(np.array(model.predict(predictionarray))))
print(firstprediction)

predictionarray[0][1][0]=(firstprediction+1)

while i < (200-2):
    y=i+2
    predictionarray[0][y][0]=(np.argmax(np.array(model.predict(predictionarray)))+1)
    i=i+1

print(predictionarray)
