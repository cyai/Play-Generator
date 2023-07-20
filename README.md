# Shakespeare Text Generation - README

This project uses a recurrent neural network (RNN) to generate text in the style of William Shakespeare. The RNN model is trained on Shakespeare's text, and then it can generate new text based on a given starting string.

## Introduction

Text generation is an application of natural language processing (NLP) where the goal is to create new text that resembles a specific style or author's writing. In this project, we use a character-based RNN to generate Shakespeare-like text.
------
## Data

The dataset used in this project is a collection of William Shakespeare's works. The text is loaded and preprocessed to prepare it for training the RNN model.
------
## Code Structure

The Python code in the Jupyter Notebook is structured into different sections:

1. **Data Loading and Preprocessing**: The Shakespeare text is loaded and converted to numerical sequences for training the RNN.

2. **Model Architecture**: The RNN model architecture is defined using the Keras Sequential API. The model consists of an Embedding layer, an LSTM layer, and a Dense layer.

3. **Loss Function and Compilation**: The loss function for the RNN model is implemented, and the model is compiled with the Adam optimizer.

4. **Model Training**: The model is trained on the Shakespeare text data using the training examples and targets.

5. **Text Generation**: The trained model can generate new text based on a given starting string.
------
------
## How the Code Works

1. **Data Loading and Preprocessing**:
   - The Shakespeare text is loaded from a URL using the TensorFlow library.
   - The text is then decoded into a string and its length is calculated.
   - Each unique character in the text is mapped to an index, creating a character-to-index mapping (`char2idx`) and an index-to-character mapping (`idx2char`).
   - The entire text is converted into a numerical representation using the `text_to_int` function.

2. **Model Architecture**:
   - The RNN model is built using the Keras Sequential API.
   - It consists of an Embedding layer, an LSTM layer with 1024 units, and a Dense layer with the same number of units as the vocabulary size (65, representing each unique character).
   - The model is designed to take sequences of characters as input and predict the next character in the sequence.

3. **Loss Function and Compilation**:
   - The loss function is implemented as a sparse categorical cross-entropy loss, as the output of the model is integer-encoded characters.
   - The model is compiled with the Adam optimizer and the defined loss function.

4. **Model Training**:
   - The dataset is shuffled and batched to create training examples and targets.
   - The model is trained on the training dataset for a set number of epochs, and checkpoints are saved during training.

5. **Text Generation**:
   - After training, the model can be used to generate text based on a given starting string.
   - The `generate_text` function takes a starting string as input and generates new text using the trained RNN model.
   - The generated text is obtained by predicting the next character based on the previous characters, and this process is repeated iteratively.


------
------
## How to Use the Code

1. Clone or download the repository to your local machine.

2. Install the required libraries (Keras, TensorFlow, and NumPy).

3. Open the Jupyter Notebook `Shakespeare_Text_Generation.ipynb`.

4. Follow the code step-by-step to understand how the RNN model is constructed, trained, and used for text generation.

5. To generate text, run the last code cell and enter a starting string when prompted.

------
------
## Dependencies

To run this code, you need the following dependencies:

- Python 3.9.0
- Jupyter Notebook
- TensorFlow 2.x
- Keras
- NumPy



Feel free to experiment with different starting strings to see how the model generates text in Shakespeare's style!

## Credits

The Shakespeare text dataset used in this project is sourced from the TensorFlow library. The RNN model architecture is inspired by research in Natural Language Processing and Text Generation.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

Please feel free to reach out for any questions or feedback regarding the project! Happy coding!
