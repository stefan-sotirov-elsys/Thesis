import matplotlib.pyplot as plt
import tensorflow as tf

from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.models import Sequential

# prepare the dataset

data_dir = "../../dataset/"

width = 224

height = 224

batch_size = 32

training_dataset = tf.keras.utils.image_dataset_from_directory(
	data_dir,
	validation_split = 0.2,
	subset = "training",
	seed = 123,
	image_size = (width, height),
	batch_size = batch_size
)

validation_dataset = tf.keras.utils.image_dataset_from_directory(
	data_dir,
	validation_split = 0.2,
	subset = "validation",
	seed = 123,
	image_size = (width, height),
	batch_size = batch_size
)

test_dataset = validation_dataset.take(
	tf.data.experimental.cardinality(validation_dataset)
) 

class_names = training_dataset.class_names

# print(class_names) # debug

# optimise the dataset

training_dataset = training_dataset.cache().shuffle(1000).prefetch(buffer_size = tf.data.AUTOTUNE)

validation_dataset = validation_dataset.cache().prefetch(buffer_size = tf.data.AUTOTUNE)

test_dataset = test_dataset.prefetch(buffer_size = tf.data.AUTOTUNE)

data_augmentation = keras.Sequential(
	[
		layers.RandomFlip("horizontal",
		input_shape = (width, height, 3)),
		layers.RandomRotation(0.1),
		layers.RandomZoom(0.1),
	]
)

# create the model

model = Sequential([
	data_augmentation,
	layers.experimental.preprocessing.Rescaling(1 / 255),
	layers.Conv2D(16, 3, padding = 'same', activation = 'relu'),
	layers.MaxPooling2D(),
	layers.Conv2D(32, 3, padding = 'same', activation = 'relu'),
	layers.MaxPooling2D(),
	layers.Conv2D(64, 3, padding = 'same', activation = 'relu'),
	layers.MaxPooling2D(),
	layers.Dropout(0.2),
	layers.Flatten(),
	layers.Dense(1)
])

early_stop_callback = callback = tf.keras.callbacks.EarlyStopping(monitor = 'loss', patience = 3)

# compile the model

model.compile(
	optimizer = 'adam',
	loss = tf.keras.losses.BinaryCrossentropy(from_logits = True),
	metrics = ['accuracy']
)

# model.summary() # debug

# train the model

history = model.fit(
	training_dataset, 
	validation_data = validation_dataset,
	epochs = 100000,
	callbacks = [early_stop_callback]
)

metrics = model.evaluate(test_dataset)

print("test loss: " + str(metrics[0]) + "test accuracy: " + str(metrics[1]))

model.save("models/custom")