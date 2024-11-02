#Next Word Prediction with LSTM and GRU
This project implements a next-word prediction system using LSTM (Long Short-Term Memory) and GRU (Gated Recurrent Unit) models. 
The model is trained on the text of Shakespeare's "Hamlet" and can predict the next word based on a given sequence of words. 
It also includes a Streamlit application for user interaction.

Project Overview
The aim of this project is to create a predictive text model that can suggest the next word in a sentence based on prior context. The model leverages the capabilities of LSTM and GRU architectures to understand the sequential nature of text data, which is essential for natural language processing tasks. By training on the rich language of "Hamlet," the model can generate contextually relevant predictions.

Objectives
Develop a robust text prediction model using LSTM and GRU architectures.
Allow for user-friendly interaction through a Streamlit application.
Evaluate and compare the performance of LSTM and GRU models in terms of accuracy and loss.
Technologies Used
This project utilizes the following technologies:

Python: The programming language used for model implementation and data processing.
TensorFlow: A powerful library for building and training deep learning models.
Keras: An API within TensorFlow for constructing neural network architectures.
NumPy: A library for numerical computing in Python, particularly for array manipulations.
scikit-learn: A library for machine learning that aids in data splitting and evaluation.
Streamlit: A framework for building interactive web applications for data science projects.

Dataset Description
The dataset used in this project is the complete text of Shakespeare's "Hamlet." The text file, hamlet.txt, must be present in the root directory of the project. This file serves as the foundation for training the LSTM and GRU models. The text is preprocessed to remove punctuation and convert all characters to lowercase to maintain consistency.

Usage Instructions
Load and Preprocess the Dataset: The script reads the text from hamlet.txt, processes it to create sequences of words, and prepares the data for training.

Train the Models: Run the training script to train both the LSTM and GRU models. This will generate and save the trained models, which can be used for predictions.

Predict the Next Word: After training, you can input a sequence of words to get the next word prediction from the models.

Model Training
The training script does the following:

Tokenization: It tokenizes the text and creates input sequences for the model.
Padding: It pads the sequences to ensure they have uniform lengths for processing.
Training Split: The dataset is split into training and testing sets (80% training, 20% testing) to evaluate model performance.
Model Definition:
LSTM Model:
An embedding layer that converts word indices into dense vectors.
Two LSTM layers that capture temporal dependencies in the data.
A dropout layer for regularization.
A final dense layer with softmax activation to output probabilities for each word in the vocabulary.
GRU Model:
Similar structure to the LSTM model but utilizes GRU layers, which are generally more efficient in terms of computation.
Both models are trained for 17 epochs. After training, their performance is evaluated based on loss and accuracy metrics.

Future Improvements
Model Enhancements: Explore additional architectures like Transformers to improve prediction accuracy.
Data Augmentation: Utilize more extensive datasets to increase model robustness.
User Interface Enhancements: Improve the Streamlit application’s interface for a more user-friendly experience.
Hyperparameter Tuning: Implement a systematic approach to hyperparameter tuning for optimizing model performance.
Deployment: Consider deploying the application on platforms like Heroku or AWS for broader access.
