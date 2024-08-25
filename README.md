Plant Disease Detection System
This project is a deep learning-based system for detecting and classifying plant diseases using images of leaves. The goal is to assist farmers and agricultural professionals in identifying plant diseases early, enabling timely intervention to improve crop yield and quality.

Project Structure
Dataset: The dataset used in this project is a collection of plant leaf images categorized into three classes: Healthy, Powdery, and Rust. The dataset is divided into training, validation, and testing sets.

Train/Train/: Contains images used for training the model.
Validation/Validation/: Contains images used for validating the model.
Test/Test/: Contains images used for testing the model.
Model: The project uses a convolutional neural network (CNN) built with TensorFlow and Keras. The model architecture includes multiple layers such as convolutional, pooling, and fully connected layers, optimized for image classification tasks.

Image Preprocessing:

ImageDataGenerator: Used for augmenting the training data with techniques like rescaling, zooming, shearing, and horizontal flipping. This helps in improving the model's generalization ability.
The images are rescaled to normalize the pixel values between 0 and 1.
Training:

The model is trained on the augmented dataset using categorical cross-entropy loss and the Adam optimizer.
The training process involves monitoring the validation loss and accuracy to prevent overfitting.
Testing:

The trained model is evaluated on the test dataset to assess its performance in real-world scenarios.
Accuracy, precision, recall, and F1-score metrics are used for performance evaluation.
Files and Directories
train_model.py: Script for training the CNN model.
preprocess_data.py: Script for preprocessing and augmenting the dataset using ImageDataGenerator.
plant_disease_model.h5: The saved model file after training.
README.md: Project documentation (this file).
app.py: Flask-based web application for deploying the model (if applicable).
requirements.txt: Lists the dependencies required to run the project.


Contributions
Contributions to this project are welcome. If you find a bug or have a feature request, please open an issue or submit a pull request.

License
This project is licensed under the MIT License - see the LICENSE file for details.
