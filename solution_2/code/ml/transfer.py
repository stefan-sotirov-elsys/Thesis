import tensorflow as tf

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

class_names = training_dataset.class_names

# optimise the dataset

training_dataset = training_dataset.cache().shuffle(1000).prefetch(buffer_size = tf.data.AUTOTUNE)

validation_dataset = validation_dataset.cache().prefetch(buffer_size = tf.data.AUTOTUNE)

# create the base model

base_model = tf.keras.applications.MobileNetV2(
	input_shape = (width, height, 3),
	include_top = False,
	weights = "imagenet"
)

# freeze the base model / unfreeze it

base_model.trainable = True

# Fine-tune from this layer onwards

fine_tune_start = 134

# Freeze all the layers before the fine_tune_start layer

for layer in base_model.layers[:fine_tune_start]:

    layer.trainable = False

print(len(base_model.layers))

# add the classification head

data_augmentation = tf.keras.Sequential([
	tf.keras.layers.RandomFlip("horizontal"),
	tf.keras.layers.RandomRotation(0.2),
	tf.keras.layers.RandomZoom(0.1),
])

global_avg_layer = tf.keras.layers.GlobalAveragePooling2D()

prediction_layer = tf.keras.layers.Dense(3)

input = tf.keras.Input(shape = (width, height, 3))

augmented_input = data_augmentation(input)

preprocessed_input = tf.keras.applications.mobilenet_v2.preprocess_input(augmented_input)

base_model_output = base_model(preprocessed_input, training = True)

avg_layer_output = global_avg_layer(base_model_output)

avg_layer_output_dropout = tf.keras.layers.Dropout(0.2)(avg_layer_output)

output = prediction_layer(avg_layer_output_dropout)

model = tf.keras.Model(input, output)

# compile the model

base_learning_rate = 0.0001

early_stop_callback = tf.keras.callbacks.EarlyStopping(monitor = 'loss', patience = 3)

model.compile(
	loss = tf.keras.losses.SparseCategoricalCrossentropy(from_logits = True),
	optimizer = tf.keras.optimizers.Adam(learning_rate = base_learning_rate / 10),
	metrics = ["accuracy"]
)

# train the model

history = model.fit(
	training_dataset,
	epochs = 100000,
	validation_data = validation_dataset,
	callbacks = [early_stop_callback]
)

model.save("models/transfer")