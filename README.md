# NLP Assignment

The following versions of packages need to be installed to accurately reproduce my results.

```
pip install tensorflow-gpu==2.7.0 nltk==3.5 numpy==1.19.2 scikit-learn==0.23.2
```

# Preprocessing

All the preprocessing is taken care of in the "nlp_assigment.py" script (the same script also defines, trains, and tests the models). All lines in the e-mails with the following characters are removed (along with empty lines):

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

A Support Vector Machine is chosen to be a baseline model. TF-IDF features were chosen to be fed to the SVM. TF-IDF stands for term frequency–inverse document frequency. It shows the importance of a word in a given document. Scikit-learn was used for both TF-IDF and SVM.

## Recurrent Neural Network Model

An LSTM model was chosen to model the classification problem given the emails. Pre-trained embeddings from the GloVe model was downloaded from [here](http://nlp.stanford.edu/data/glove.6B.zip). I used the 50-dimensional embeddings. For words in GloVe's vocabulary, the embeddings were initialized with the saved vectors, otherwise they were initialized randomly. All the embeddings are, however, trained (it can be considered a fine-tuning of existing embeddings). The overall architecture is as follows:

<p align="center">
<img src="https://user-images.githubusercontent.com/76472410/174401621-51cad782-ac3d-466c-8620-4c82e5026e44.png" width="400">
</p>

Tensorflow was used to construct and train the model.

# Results

The LSTM model performs better than the SVM in terms of accuracy, f1-score, and precision. However, SVM has a higher Recall.

<p align="center">
<img src="https://user-images.githubusercontent.com/76472410/174397284-e14c32be-46f2-421c-9f91-c89f91906c99.png" width="400">
<img src="https://user-images.githubusercontent.com/76472410/174404491-57d7bc9d-500d-4cb1-8d35-63ec5cf2979d.png" width="400">
<img src="https://user-images.githubusercontent.com/76472410/174397335-0573a6d3-34c9-46d1-ba2c-1066bd77d28c.png" width="400">
<img src="https://user-images.githubusercontent.com/76472410/174397346-ee53922c-c93b-4208-bd8c-d75da6edb000.png" width="400">
<img src="https://user-images.githubusercontent.com/76472410/174397361-67352d74-fe00-471f-a178-83e0964f6f9a.png" width="400">
</p>

The loss plot also shows that the training of the LSTM model was fairly stable. However, the model seems to starting to overfit towards the end. This is expected due to the large number of parameters in the model. The model could possibly perform better if its hyperparameters are tuned using a validation, especially to alleviate overfitting.
