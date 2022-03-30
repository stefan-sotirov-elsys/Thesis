import tensorflow as tf

# prepare the dataset

data_dir = "../../datasets/training/"

width = 224

height = 224

batch_size = 32

training_dataset = tf.keras.utils.image_dataset_from_directory(
	data_dir,
	label_mode = "binary",
	class_names = ["clean", "dirty"],
	validation_split = 0.2,
	subset = "validation",
	seed = 123,
	image_size = (width, height),
	batch_size = batch_size
)

validation_dataset = tf.keras.utils.image_dataset_from_directory(
	data_dir,
	label_mode = "binary",
	class_names = ["clean", "dirty"],
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

data_augmentation = tf.keras.models.Sequential([
		tf.keras.layers.RandomFlip("horizontal",
			input_shape = (width, height, 3)),
		tf.keras.layers.RandomRotation(0.1),
		tf.keras.layers.RandomZoom(0.1),
])

# create the model

model = tf.keras.models.Sequential([
	data_augmentation,
	tf.keras.layers.experimental.preprocessing.Rescaling(1 / 255),
	tf.keras.layers.Conv2D(16, 3, padding = "same", activation = "leaky_relu"),
	tf.keras.layers.MaxPooling2D(),
	tf.keras.layers.Conv2D(32, 3, padding = "same", activation = "leaky_relu"),
	tf.keras.layers.MaxPooling2D(),
	tf.keras.layers.Conv2D(64, 3, padding = "same", activation = "leaky_relu"),
	tf.keras.layers.MaxPooling2D(),
	tf.keras.layers.Dropout(0.2),
	tf.keras.layers.Flatten(),
#	tf.keras.layers.Dense(128, activation = 'leaky_relu'),
	tf.keras.layers.Dense(1, activation = "sigmoid")
])

early_stop_callback = callback = tf.keras.callbacks.EarlyStopping(monitor = "loss", patience = 3)

# compile the model

model.compile(
	optimizer = "adam",
	loss = tf.keras.losses.BinaryCrossentropy(from_logits = False),
	metrics = [
		tf.keras.metrics.FalseNegatives(),
		tf.keras.metrics.FalsePositives(),
		tf.keras.metrics.TrueNegatives(),
		tf.keras.metrics.TruePositives(),
		"accuracy"
	]
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

model.save("models/custom")