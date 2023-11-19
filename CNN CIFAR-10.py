######################################## ujjwal Kumar ###################################################################
################### 22MT10061 ##############################################################################################
################# ASSIGnment 6 #################################################################################


############## Implementaion ##########################################################################

################## CNN VANILLA ####################################################################################
# Import necessary libraries
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

# Define the CNN model (CNN-Vanilla)
model = keras.Sequential([
    layers.Conv2D(32, (3, 3), activation='relu', input_shape=(64, 64, 3)),
    layers.MaxPooling2D((2, 2)),
    layers.Conv2D(64, (3, 3), activation='relu'),
    layers.MaxPooling2D((2, 2)),
    layers.Conv2D(128, (3, 3), activation='relu'),
    layers.MaxPooling2D((2, 2)),
    layers.Flatten(),
    layers.Dense(128, activation='relu'),
    layers.Dense(3, activation='softmax')
])

# Compile the model
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Print model summary
model.summary()

################ CNN Resnet ###########################################################################

# Define the CNN-Resnet model with residual connections
def residual_block(x, filters, kernel_size=3, stride=1, use_batch_norm=True):
    y = layers.Conv2D(filters, kernel_size=kernel_size, strides=stride, padding='same')(x)
    if use_batch_norm:
        y = layers.BatchNormalization()(y)
    y = layers.Activation('relu')(y)
    y = layers.Conv2D(filters, kernel_size=kernel_size, strides=stride, padding='same')(y)
    if use_batch_norm:
        y = layers.BatchNormalization()(y)
    # Adjust the dimensions of the input tensor to match the output tensor
    if x.shape[-1] != filters:
        x = layers.Conv2D(filters, kernel_size=1, strides=stride, padding='same')(x)
    out = layers.add([x, y])
    out = layers.Activation('relu')(out)
    return out

inputs = keras.Input(shape=(64, 64, 3))
x = layers.Conv2D(32, (3, 3), activation='relu', padding='same')(inputs)
x = layers.MaxPooling2D((2, 2))(x)

# Create residual blocks
num_blocks = 3  # You can adjust the number of residual blocks
for _ in range(num_blocks):
    x = residual_block(x, 64)

x = layers.GlobalAveragePooling2D()(x)
x = layers.Dense(128, activation='relu')(x)
outputs = layers.Dense(3, activation='softmax')(x)

model_resnet = keras.Model(inputs, outputs)

# Compile the models
model_resnet.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Print model summaries
model_resnet.summary()

########################################## EXPERIMENT 1 ############################################################################

### EXP 1
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.datasets import cifar10

# Load CIFAR-10 dataset
(train_images, train_labels), (test_images, test_labels) = cifar10.load_data()

# Preprocess and normalize the data
train_images = train_images / 255.0
test_images = test_images / 255.0

# Define the CNN-Vanilla model
model_vanilla = keras.Sequential([
    layers.Conv2D(32, (3, 3), activation='relu', input_shape=(32, 32, 3)),
    layers.MaxPooling2D((2, 2)),
    layers.Conv2D(64, (3, 3), activation='relu'),
    layers.MaxPooling2D((2, 2)),
    layers.Conv2D(128, (3, 3), activation='relu'),
    layers.MaxPooling2D((2, 2)),
    layers.Flatten(),
    layers.Dense(128, activation='relu'),
    layers.Dense(10, activation='softmax')  # Use 10 units for multi-class classification
])

# Define the CNN-Resnet model with residual connections
def residual_block(x, filters, kernel_size=3, stride=1, use_batch_norm=True):
    y = layers.Conv2D(filters, kernel_size=kernel_size, strides=stride, padding='same')(x)
    if use_batch_norm:
        y = layers.BatchNormalization()(y)
    y = layers.Activation('relu')(y)
    y = layers.Conv2D(filters, kernel_size=kernel_size, strides=stride, padding='same')(y)
    if use_batch_norm:
        y = layers.BatchNormalization()(y)
    # Adjust the dimensions of the input tensor to match the output tensor
    if x.shape[-1] != filters:
        x = layers.Conv2D(filters, kernel_size=1, strides=stride, padding='same')(x)
    out = layers.add([x, y])
    out = layers.Activation('relu')(out)
    return out

inputs = keras.Input(shape=(32, 32, 3))
x = layers.Conv2D(32, (3, 3), activation='relu', padding='same')(inputs)
x = layers.MaxPooling2D((2, 2))(x)

# Create residual blocks
num_blocks = 3  # You can adjust the number of residual blocks
for _ in range(num_blocks):
    x = residual_block(x, 64)

x = layers.GlobalAveragePooling2D()(x)
x = layers.Dense(128, activation='relu')(x)
outputs = layers.Dense(10, activation='softmax')(x)  # Use 10 units for multi-class classification

model_resnet = keras.Model(inputs, outputs)

# Set the learning rate to 0.001 for both models using the legacy optimizer
learning_rate = 0.001
optimizer = tf.keras.optimizers.legacy.Adam(learning_rate=learning_rate)

# Compile the models with the specified learning rate
model_vanilla.compile(optimizer=optimizer, loss='sparse_categorical_crossentropy', metrics=['accuracy'])
model_resnet.compile(optimizer=optimizer, loss='sparse_categorical_crossentropy', metrics=['accuracy'])



# Train CNN-Resnet
model_resnet.fit(train_images, train_labels, epochs=30, batch_size=32, verbose=2)
# Train CNN-Vanilla
model_vanilla.fit(train_images, train_labels, epochs=30, batch_size=32, verbose=2)


# Evaluate both models on the test set
accuracy_vanilla = model_vanilla.evaluate(test_images, test_labels, verbose=2)[1] * 100
accuracy_resnet = model_resnet.evaluate(test_images, test_labels, verbose=2)[1] * 100

# Print accuracy
print("CNN-Vanilla Accuracy on Test Set: {}%".format(accuracy_vanilla))
print("CNN-Resnet Accuracy on Test Set: {}%".format(accuracy_resnet))

# Print the number of parameters in each model
num_params_vanilla = model_vanilla.count_params()
num_params_resnet = model_resnet.count_params()
print("Number of Parameters in CNN-Vanilla: {}".format(num_params_vanilla))
print("Number of Parameters in CNN-Resnet: {}".format(num_params_resnet))

# Determine the best choice among the two models
best_choice = 'CNN-Vanilla' if accuracy_vanilla > accuracy_resnet else 'CNN-Resnet'
print(f"The best choice among the two models is: {best_choice}")


import matplotlib.pyplot as plt

# Train CNN-Resnet
history_resnet = model_resnet.fit(train_images, train_labels, epochs=10, batch_size=256, verbose=2)

# Train CNN-Vanilla
history_vanilla = model_vanilla.fit(train_images, train_labels, epochs=10, batch_size=256, verbose=2)

# Plot training accuracy vs. epochs
plt.figure(figsize=(12, 6))
plt.plot(history_vanilla.history['accuracy'], label='CNN-Vanilla', marker='o')
plt.plot(history_resnet.history['accuracy'], label='CNN-Resnet', marker='x')
plt.title('Training Accuracy vs. Epochs')
plt.xlabel('Epochs')
plt.ylabel('Training Accuracy')
plt.legend()
plt.grid()
plt.show()
##############################################################################################################################################

##### EXPERIMENT 2 ####################################################################################################

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.datasets import cifar10

# Load CIFAR-10 dataset
(train_images, train_labels), (test_images, test_labels) = cifar10.load_data()

# Preprocess and normalize the data (with data normalization)
train_images_normalized = train_images / 255.0
test_images_normalized = test_images / 255.0

# Define the CNN-Vanilla model
model_vanilla = keras.Sequential([
    layers.Conv2D(32, (3, 3), activation='relu', input_shape=(32, 32, 3)),
    layers.MaxPooling2D((2, 2)),
    layers.Conv2D(64, (3, 3), activation='relu'),
    layers.MaxPooling2D((2, 2)),
    layers.Conv2D(128, (3, 3), activation='relu'),
    layers.MaxPooling2D((2, 2)),
    layers.Flatten(),
    layers.Dense(128, activation='relu'),
    layers.Dense(10, activation='softmax')  # Use 10 units for multi-class classification
])

# Set the learning rate to 0.001 using the legacy optimizer
learning_rate = 0.001
optimizer = tf.keras.optimizers.legacy.Adam(learning_rate=learning_rate)

# Compile the model with the specified learning rate
model_vanilla.compile(optimizer=optimizer, loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# Train CNN-Vanilla with data normalization
model_vanilla.fit(train_images_normalized, train_labels, epochs=10, batch_size=256, verbose=2)

# Train CNN-Vanilla without data normalization (using the original data)
model_vanilla_without_normalization = keras.Sequential([
    layers.Conv2D(32, (3, 3), activation='relu', input_shape=(32, 32, 3)),
    layers.MaxPooling2D((2, 2)),
    layers.Conv2D(64, (3, 3), activation='relu'),
    layers.MaxPooling2D((2, 2)),
    layers.Conv2D(128, (3, 3), activation='relu'),
    layers.MaxPooling2D((2, 2)),
    layers.Flatten(),
    layers.Dense(128, activation='relu'),
    layers.Dense(10, activation='softmax')
])

# Compile the model with the specified learning rate
model_vanilla_without_normalization.compile(optimizer=optimizer, loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# Train CNN-Vanilla without data normalization (using the original data)
model_vanilla_without_normalization.fit(train_images, train_labels, epochs=10, batch_size=256, verbose=2)

# Evaluate both models on the test set
accuracy_vanilla_with_normalization = model_vanilla.evaluate(test_images_normalized, test_labels, verbose=2)[1] * 100
accuracy_vanilla_original = model_vanilla_without_normalization.evaluate(test_images, test_labels, verbose=2)[1] * 100

# Print accuracy
print("CNN-Vanilla Accuracy with Data Normalization: {}%".format(accuracy_vanilla_with_normalization))
print("CNN-Vanilla Accuracy without Data Normalization: {}%".format(accuracy_vanilla_original))

##################################################################### done with exp 2 ##########################

################################## Experiment 3 ###################################################################################

### EXP 3
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.datasets import cifar10

# Load CIFAR-10 dataset
(train_images, train_labels), (test_images, test_labels) = cifar10.load_data()

# Preprocess and normalize the data (with data normalization)
train_images_normalized = train_images / 255.0
test_images_normalized = test_images / 255.0

# Define the CNN-Vanilla model
model_vanilla = keras.Sequential([
    layers.Conv2D(32, (3, 3), activation='relu', input_shape=(32, 32, 3)),
    layers.MaxPooling2D((2, 2)),
    layers.Conv2D(64, (3, 3), activation='relu'),
    layers.MaxPooling2D((2, 2)),
    layers.Conv2D(128, (3, 3), activation='relu'),
    layers.MaxPooling2D((2, 2)),
    layers.Flatten(),
    layers.Dense(128, activation='relu'),
    layers.Dense(10, activation='softmax')  # Use 10 units for multi-class classification
])

# Set the learning rate to 0.001 using the legacy optimizer
learning_rate = 0.001
batch_size = 256
epochs = 10

# Create a dictionary of optimizers to study
optimizers = {
    "Stochastic Gradient Descent": tf.keras.optimizers.SGD(learning_rate=learning_rate),
    "Mini-batch Gradient Descent (No Momentum)": tf.keras.optimizers.SGD(learning_rate=learning_rate, momentum=0.0),
    "Mini-batch Gradient Descent (Momentum 0.9)": tf.keras.optimizers.SGD(learning_rate=learning_rate, momentum=0.9),
    "ADAM Optimizer": tf.keras.optimizers.Adam(learning_rate=learning_rate)
}

# Train and evaluate the model with each optimizer
results = {}
for optimizer_name, optimizer in optimizers.items():
    model_vanilla.compile(optimizer=optimizer, loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    model_vanilla.fit(train_images_normalized, train_labels, epochs=epochs, batch_size=batch_size, verbose=2)
    accuracy = model_vanilla.evaluate(test_images_normalized, test_labels, verbose=2)[1] * 100
    results[optimizer_name] = accuracy

# Print the results
for optimizer_name, accuracy in results.items():
    print(f"{optimizer_name} Accuracy with Data Normalization: {accuracy}%")
    
################################################################### done with experiment   ############################################


############################### EXPERIMENT 4 ################################################################################

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.datasets import cifar10

# Load CIFAR-10 dataset
(train_images, train_labels), (test_images, test_labels) = cifar10.load_data()

# Preprocess and normalize the data
train_images = train_images / 255.0
test_images = test_images / 255.0

# Define the residual_block function
def residual_block(x, filters, kernel_size=3, stride=1, use_batch_norm=True):
    y = layers.Conv2D(filters, kernel_size=kernel_size, strides=stride, padding='same')(x)
    if use_batch_norm:
        y = layers.BatchNormalization()(y)
    y = layers.Activation('relu')(y)
    y = layers.Conv2D(filters, kernel_size=kernel_size, strides=stride, padding='same')(y)
    if use_batch_norm:
        y = layers.BatchNormalization()(y)
    # Adjust the dimensions of the input tensor to match the output tensor
    if x.shape[-1] != filters:
        x = layers.Conv2D(filters, kernel_size=1, strides=stride, padding='same')(x)
    out = layers.add([x, y])
    out = layers.Activation('relu')(out)
    return out

# Define the CNN-Resnet model with varying network depth
def create_resnet_model(num_resnet_blocks, num_fc_layers):
    inputs = keras.Input(shape=(32, 32, 3))
    x = layers.Conv2D(32, (3, 3), activation='relu', padding='same')(inputs)
    x = layers.MaxPooling2D((2, 2))(x)

    # Create residual blocks
    for _ in range(num_resnet_blocks):
        x = residual_block(x, 64)

    # Add fully connected layers
    for _ in range(num_fc_layers):
        x = layers.Dense(128, activation='relu')(x)

    outputs = layers.Dense(10, activation='softmax')(x)  # Use 10 units for multi-class classification

    return keras.Model(inputs, outputs)

# Set the learning rate and other parameters
learning_rate = 0.001
batch_size = 256
epochs = 10

# Define the network configurations to study
network_configs = [
    {"num_resnet_blocks": 4, "num_fc_layers": 2},  # (a) Four level Resnet block with two fully-connected layers
    {"num_resnet_blocks": 3, "num_fc_layers": 4},  # (b) Three level Resnet blocks with four fully-connected layers
]

# Train and evaluate the models for different network configurations
results = {}
for config in network_configs:
    model_resnet = create_resnet_model(config["num_resnet_blocks"], config["num_fc_layers"])
    model_resnet.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
                         loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    model_resnet.fit(train_images, train_labels, epochs=epochs, batch_size=batch_size, verbose=2)
    accuracy = model_resnet.evaluate(test_images, test_labels, verbose=2)[1] * 100
    num_params = model_resnet.count_params()
    results[f"Resnet ({config['num_resnet_blocks']} Blocks, {config['num_fc_layers']} FC Layers)"] = {
        "Accuracy": accuracy,
        "Number of Parameters": num_params
    }

# Print the results
for network_name, metrics in results.items():
    print(f"{network_name} Accuracy: {metrics['Accuracy']:.2f}%")
    print(f"Number of Parameters: {metrics['Number of Parameters']}")
    
#### done with all the experiment #########################################################################################