# Data Preprocessing
import numpy as np
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, LSTM, Dense, Dropout, GRU
from sklearn.model_selection import train_test_split

# Load and preprocess the dataset
with open('hamlet.txt', 'r') as file:
    text = file.read().lower()

# Tokenize the text
tokenizer = Tokenizer()
tokenizer.fit_on_texts([text])
total_words = len(tokenizer.word_index) + 1  # Vocabulary size
print("Total words:", total_words)  # Sanity check

# Create input sequences
input_sequences = []
for line in text.split('\n'):
    token_list = tokenizer.texts_to_sequences([line])[0]
    for i in range(1, len(token_list)):
        n_gram_sequence = token_list[:i + 1]
        input_sequences.append(n_gram_sequence)

# Pad sequences to ensure equal length
max_sequence_len = max([len(x) for x in input_sequences])
print("Max sequence length:", max_sequence_len)  # Sanity check
input_sequences = np.array(pad_sequences(input_sequences, maxlen=max_sequence_len, padding='pre'))

# Create predictors and labels
x, y = input_sequences[:, :-1], input_sequences[:, -1]
y = tf.keras.utils.to_categorical(y, num_classes=total_words)

# Split the data into training and testing sets
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2)

# Define the LSTM model
model_lstm = Sequential([
    Embedding(input_dim=total_words, output_dim=100),
    LSTM(150, return_sequences=True),
    Dropout(0.2),
    LSTM(100),
    Dense(total_words, activation="softmax")
])

# Compile the LSTM model
model_lstm.compile(loss="categorical_crossentropy", optimizer='adam', metrics=['accuracy'])

# Train the LSTM model
history_lstm = model_lstm.fit(x_train, y_train, epochs=17, validation_data=(x_test, y_test), verbose=1)

# Evaluate the LSTM model
lstm_loss, lstm_accuracy = model_lstm.evaluate(x_test, y_test, verbose=0)
print(f"LSTM Model - Loss: {lstm_loss}, Accuracy: {lstm_accuracy}")

# Define the GRU model
model_gru = Sequential([
    Embedding(input_dim=total_words, output_dim=100),
    GRU(150, return_sequences=True),
    Dropout(0.2),
    GRU(100),
    Dense(total_words, activation="softmax")
])

# Compile the GRU model
model_gru.compile(loss="categorical_crossentropy", optimizer='adam', metrics=['accuracy'])

# Train the GRU model
history_gru = model_gru.fit(x_train, y_train, epochs=17, validation_data=(x_test, y_test), verbose=1)

# Evaluate the GRU model
gru_loss, gru_accuracy = model_gru.evaluate(x_test, y_test, verbose=0)
print(f"GRU Model - Loss: {gru_loss}, Accuracy: {gru_accuracy}")

# Function to predict the next word
def predict_next_word(model, tokenizer, text, max_sequence_len):
    token_list = tokenizer.texts_to_sequences([text])[0]
    if len(token_list) >= max_sequence_len:
        token_list = token_list[-(max_sequence_len - 1):]  # Ensure the sequence length matches max_sequence_len-1
    token_list = pad_sequences([token_list], maxlen=max_sequence_len - 1, padding='pre')
    predicted = model.predict(token_list, verbose=0)
    predicted_word_index = np.argmax(predicted, axis=1)
    for word, index in tokenizer.word_index.items():
        if index == predicted_word_index:
            return word
    return None

# Example prediction
input_text = "to be or not to "
print(f"Input text: {input_text}")

# Get predictions from the LSTM model
next_word_lstm = predict_next_word(model_lstm, tokenizer, input_text, max_sequence_len)
print(f"LSTM Next Word Prediction: {next_word_lstm}")

# Get predictions from the GRU model
next_word_gru = predict_next_word(model_gru, tokenizer, input_text, max_sequence_len)
print(f"GRU Next Word Prediction: {next_word_gru}")

# Save the models
model_lstm.save("next_word_lstm.h5")
model_gru.save("next_word_gru.h5")

# Save the tokenizer
import pickle
with open('tokenizer.pickle', 'wb') as handle:
    pickle.dump(tokenizer, handle, protocol=pickle.HIGHEST_PROTOCOL)
