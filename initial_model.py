import pandas as pd
import os
import tensorflow as tf
import keras

print("Modules imported")





def create_dataset(pandas_dataset):
	pandas_dataset = pandas_dataset.fillna(0)

	features = pandas_dataset.loc[:, ['a0_x', 'a0_y', 'a0_z', \
									'a1_x', 'a1_y', 'a1_z', \
									'a2_x', 'a2_y', 'a2_z',
									'a3_x', 'a3_y', 'a3_z',
									],]
	labels = pandas_dataset.loc[:, ['b0_x', 'b0_y', 'b0_z', \
								  'b1_x', 'b1_y', 'b1_z',
								  'b2_x', 'b2_y', 'b2_z',
								  'b3_x', 'b3_y', 'b3_z']]

	features = features.values
	labels = labels.values

	return features, labels


def import_dataset():
	print("Beginning import of dataset")
	files = []
	for i in os.listdir(os.getcwd()):
	    if i.endswith('.csv'):
	        files.append(open(i))

	combined_csv = pd.concat( [ pd.read_csv(f) for f in files ] )
	print("Completed import of dataset")


	features, labels = create_dataset(combined_csv)
	assert features.shape[0] == labels.shape[0]



	features_placeholder = tf.placeholder(features.dtype, features.shape)
	labels_placeholder = tf.placeholder(labels.dtype, labels.shape)


	dataset = tf.data.Dataset.from_tensor_slices((features, labels))
	inputs = keras.Input(shape=(12,))  # Returns a placeholder tensor

	# A layer instance is callable on a tensor, and returns a tensor.
	x = keras.layers.Dense(64, activation='relu')(inputs)
	x = keras.layers.Dense(64, activation='relu')(x)
	predictions = keras.layers.Dense(12, activation='softmax')(x)

	# Instantiate the model given inputs and outputs.
	model = keras.Model(inputs=inputs, outputs=predictions)

	# The compile step specifies the training configuration.
	model.compile(optimizer=tf.train.RMSPropOptimizer(0.001),
	              loss='mean_squared_error',
	              metrics=['accuracy'])

	# Trains for 5 epochs
	model.fit(features, labels, batch_size=32, epochs=2)

	model_json = model.to_json()
	with open('location_model.model', 'w') as json_file:
		json_file.write(model_json)
	model.save_weights('location_model.model')

	


	#open(, 'r')





if __name__ == "__main__":
	import_dataset()


