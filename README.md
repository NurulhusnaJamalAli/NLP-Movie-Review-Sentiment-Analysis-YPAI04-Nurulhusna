# NLP Movie Review Sentiment YPAI04 Nurulhusna
 Train a deep learning model to analyze movie reviews and determine their sentiment (positive or negative). The project involves data loading, preprocessing for NLP, and building a bidirectional LSTM model. A Streamlit app is provided for easy user interaction with the trained model. This is an exercise by Nurulhusna.

 # Movie Review Sentiment Analysis

This project aims to train a deep learning model that can accurately identify the sentiment of movie reviews as either positive or negative. We will use the IMDB dataset, which contains labeled movie reviews, to build and deploy our sentiment analysis model.

### Prerequisites

Before running the scripts, make sure you have the following installed:

- Python (>=3.6)
- TensorFlow (>=2.0)
- Keras (>=2.0)
- Streamlit (>=0.84)
- Pandas (>=1.0)
- Pickle (>=4.0)

### Dataset

You can obtain the dataset needed for training the model from the following URL:

[IMDB-Dataset.csv](https://raw.githubusercontent.com/Ankit152/IMDB-sentiment-analysis/master/IMDB-Dataset.csv)

The dataset is in CSV format and can be loaded directly using the `pandas.read_csv()` function.

## Training the Model

To train the sentiment analysis model, follow these steps:

1. Clone this repository to your local machine.
2. Download the dataset from the provided URL and place it in the same directory as the python scripts.

### Step 1: Data Loading, Inspection, and Cleaning

The first script, `model_training.py`, will handle data loading, inspection, and cleaning. It will read the dataset using pandas, inspect the data, and perform any necessary cleaning or preprocessing.

### Step 2: Preprocessing for NLP

In this step, we will perform the following preprocessing steps on the text data:

- Tokenization: Splitting the text into individual words (tokens).
- Padding and Truncating: Ensuring all input sequences have the same length for model compatibility.
- Embedding: This will be part of the model, and it will learn to represent words as dense vectors.

### Step 3: Model Development

We will build a bidirectional LSTM model for sentiment analysis. This model will learn from the preprocessed text data and make predictions on new movie reviews.

### Step 4: Save Important Files

The model will be saved in a Keras-compatible format. Additionally, the Tokenizer object and LabelEncoder will be saved using the Pickle library for later use during deployment.

Run the python script.

The trained model and related files will be saved in the same directory.

## Deployment

The second script, `model_deployment.py`, contains the code to run a Streamlit app for sentiment analysis. The app will accept user text input and display the prediction result based on the user's input.

To deploy the Streamlit app, follow these steps:

1. Ensure you have installed all the required packages mentioned in the Prerequisites section.
2. Place the trained model and saved Tokenizer object and LabelEncoder in the same directory as the `model_deployment.py` script.
3. Run the Streamlit app using the following command:

```
streamlit run model_deployment.py
```

This will launch a local web server, and you can access the app in your web browser.


## Conclusion

With the completion of both `model_training.py` and `model_deployment.py`, you will have a fully functioning movie review sentiment analysis model that can be used to classify new movie reviews as positive or negative. Happy coding!

## Acknowledgments
The datasets used in this project are sourced from Kaggle. Credit goes to the original author:
 [IMDB-Dataset.csv](https://raw.githubusercontent.com/Ankit152/IMDB-sentiment-analysis/master/IMDB-Dataset.csv)

## License
This project is licensed under the [MIT License](LICENSE).
