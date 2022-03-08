import os
import tensorflow as tf

os.environ["TF_CPP_MIN_LOG_LEVEL"] = '3'

data_dir = "../../dataset/"

width = 224

height = 224

batch_size = 32

dataset = tf.keras.utils.image_dataset_from_directory(
	data_dir,
	label_mode = "binary",
	class_names = ["clean", "dirty"],
	validation_split = 0.2,
	subset = "validation",
	seed = 123,
	image_size = (width, height),
	batch_size = batch_size
)

# print(str(dataset)) # debug

custom = tf.keras.models.load_model("models/custom/")

custom.evaluate(dataset)

transfer = tf.keras.models.load_model("models/transfer/")

transfer.evaluate(dataset)