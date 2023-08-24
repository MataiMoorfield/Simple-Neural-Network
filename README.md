# Simple-Neural-Network
Simple neural network and training it using the MNIST dataset
This python code shows how to build, train, and evaluate a simple neural network model for classifying handwritten digits from the MNIST dataset using TensorFlow and Keras.

# Import necessary libraries:

- Import TensorFlow and its submodules for building and training neural networks.
- Import the MNIST dataset and necessary model-related modules.

# Load and preprocess the dataset:

- Load the MNIST dataset, which contains handwritten digit images and their corresponding labels for training and testing.
- Reshape the images to a flat format of 28x28 pixels and normalize the pixel values to the range [0, 1].

# Convert labels to one-hot encoding:

- Convert the integer labels to one-hot encoded vectors. One-hot encoding is a binary representation of each label, where the index of the true value represents the label.

# Build the neural network model:

- Define a sequential model, a linear stack of layers, using the Sequential class.
- Add a Flatten layer to convert the 28x28 pixel images into a flat vector.
- Add a fully connected (Dense) layer with 128 units and ReLU activation function.
- Add another Dense layer with 10 units and softmax activation function, which outputs probabilities for each class.

# Compile the model:

- Compile the model using the compile method.
- Specify the optimizer (Adam), loss function (categorical cross-entropy), and evaluation metric (accuracy).

# Train the model:

- Train the model using the fit method.
- Provide the training images and labels, specify the number of training epochs, batch size, and validation split.
- During training, the model adjusts its weights to minimize the loss function.

# Evaluate the model:

- Evaluate the trained model's performance using the evaluate method.
- Provide the test images and labels.
- Calculate the test loss and test accuracy.
- Print the test accuracy.
