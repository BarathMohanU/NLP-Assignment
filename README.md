# NLP Assignment

# Preprocessing

All the preprocessing is taken care of in the "nlp_assigment.py" script. All lines in the e-mails with the following characters are removed (along with empty lines):

```
'--', ':', '@', '<', '>'
```

Lines with these symbols are likely to be metadata like sender's address, receiver's address, etc. Following this, the lines are tokenized, stopwords are removed, words are lemmatized, all special characters and numbers are removed, and all alphabets are converted to lower case using the following packages:

```
import nltk
nltk.download('punkt')
nltk.download('wordnet')
nltk.download('stopwords')
import re
```

Top 5000 most occuring words are kept and others are removed.

# Models

## SVM with TF-IDF

A Support Vector Machine is chosen to be a baseline model. TF-IDF features were chosen to be fed to the SVM. TF-IDF stands for term frequency–inverse document frequency. It shows the importance of a word in a given document.

## Recurrent Neural Network

An LSTM model was chosen to model the classification problem given the emails. Pre-trained embeddings from the GloVe model was downloaded from [here](https://keras.io/examples/nlp/pretrained_word_embeddings/#:~:text=download%20pre%2Dtrained-,GloVe,-embeddings%20(a%20822M)

# Results

<p align="center">
<img src="https://user-images.githubusercontent.com/76472410/174397284-e14c32be-46f2-421c-9f91-c89f91906c99.png" width="512">
<img src="https://user-images.githubusercontent.com/76472410/174397308-a5fc9a83-3cf9-4d43-bb29-9fecd057b771.png" width="512">
<img src="https://user-images.githubusercontent.com/76472410/174397335-0573a6d3-34c9-46d1-ba2c-1066bd77d28c.png" width="512">
<img src="https://user-images.githubusercontent.com/76472410/174397346-ee53922c-c93b-4208-bd8c-d75da6edb000.png" width="512">
<img src="https://user-images.githubusercontent.com/76472410/174397361-67352d74-fe00-471f-a178-83e0964f6f9a.png" width="512">
</p>
