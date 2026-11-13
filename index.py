import time
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Input, Embedding, LSTM, Dense



faqs = """About the Program
What is the course fee for  Data Science Mentorship Program (DSMP 2023)
The course follows a monthly subscription model where you have to make monthly payments of Rs 799/month.
What is the total duration of the course?
The total duration of the course is 7 months. So the total course fee becomes 799*7 = Rs 5600(approx.)
What is the syllabus of the mentorship program?
We will be covering the following modules:
Python Fundamentals
Python libraries for Data Science
Data Analysis
SQL for Data Science
Maths for Machine Learning
ML Algorithms
Practical ML
MLOPs
Case studies
You can check the detailed syllabus here - https://learnwith.campusx.in/courses/CampusX-Data-Science-Mentorship-Program-637339afe4b0615a1bbed390
Will Deep Learning and NLP be a part of this program?
No, NLP and Deep Learning both are not a part of this program’s curriculum.
What if I miss a live session? Will I get a recording of the session?
Yes all our sessions are recorded, so even if you miss a session you can go back and watch the recording.
Where can I find the class schedule?
Checkout this google sheet to see month by month time table of the course - https://docs.google.com/spreadsheets/d/16OoTax_A6ORAeCg4emgexhqqPv3noQPYKU7RJ6ArOzk/edit?usp=sharing.
What is the time duration of all the live sessions?
Roughly, all the sessions last 2 hours.
What is the language spoken by the instructor during the sessions?
Hinglish
How will I be informed about the upcoming class?
You will get a mail from our side before every paid session once you become a paid user.
Can I do this course if I am from a non-tech background?
Yes, absolutely.
I am late, can I join the program in the middle?
Absolutely, you can join the program anytime.
If I join/pay in the middle, will I be able to see all the past lectures?
Yes, once you make the payment you will be able to see all the past content in your dashboard.
Where do I have to submit the task?
You don't have to submit the task. We will provide you with the solutions, you have to self evaluate the task yourself.
Will we do case studies in the program?
Yes.
Where can we contact you?
You can mail us at nitish.campusx@gmail.com
Payment/Registration related questions
Where do we have to make our payments? Your YouTube channel or website?
You have to make all your monthly payments on our website. Here is the link for our website - https://learnwith.campusx.in/
Can we pay the entire amount of Rs 5600 all at once?
Unfortunately no, the program follows a monthly subscription model.
What is the validity of monthly subscription? Suppose if I pay on 15th Jan, then do I have to pay again on 1st Feb or 15th Feb
15th Feb. The validity period is 30 days from the day you make the payment. So essentially you can join anytime you don’t have to wait for a month to end.
What if I don't like the course after making the payment. What is the refund policy?
You get a 7 days refund period from the day you have made the payment.
I am living outside India and I am not able to make the payment on the website, what should I do?
You have to contact us by sending a mail at nitish.campusx@gmail.com
Post registration queries
Till when can I view the paid videos on the website?
This one is tricky, so read carefully. You can watch the videos till your subscription is valid. Suppose you have purchased subscription on 21st Jan, you will be able to watch all the past paid sessions in the period of 21st Jan to 20th Feb. But after 21st Feb you will have to purchase the subscription again.
But once the course is over and you have paid us Rs 5600(or 7 installments of Rs 799) you will be able to watch the paid sessions till Aug 2024.
Why lifetime validity is not provided?
Because of the low course fee.
Where can I reach out in case of a doubt after the session?
You will have to fill a google form provided in your dashboard and our team will contact you for a 1 on 1 doubt clearance session
If I join the program late, can I still ask past week doubts?
Yes, just select past week doubt in the doubt clearance google form.
I am living outside India and I am not able to make the payment on the website, what should I do?
You have to contact us by sending a mail at nitish.campusx@gmai.com
Certificate and Placement Assistance related queries
What is the criteria to get the certificate?
There are 2 criterias:
You have to pay the entire fee of Rs 5600
You have to attempt all the course assessments.
I am joining late. How can I pay payment of the earlier months?
You will get a link to pay fee of earlier months in your dashboard once you pay for the current month.
I have read that Placement assistance is a part of this program. What comes under Placement assistance?
This is to clarify that Placement assistance does not mean Placement guarantee. So we dont guarantee you any jobs or for that matter even interview calls. So if you are planning to join this course just for placements, I am afraid you will be disappointed. Here is what comes under placement assistance
Portfolio Building sessions
Soft skill sessions
Sessions with industry mentors
Discussion on Job hunting strategies
what is a neural network
a neural network is a model made of layers of connected neurons
each neuron multiplies its inputs by weights and adds a bias
the neuron then applies an activation function to the sum
the input layer receives the data and passes it to the hidden layer
the hidden layer learns useful features from the input data
the output layer produces the final prediction of the model
a weight decides how strongly one neuron affects the next neuron
a bias lets a neuron shift its output up or down
the sigmoid function squeezes any number into a value between zero and one
the tanh function squeezes any number into a value between minus one and one
the relu function keeps positive numbers and turns negative numbers into zero
what does it mean to train a neural network
training a neural network means finding good values for all the weights
what is the loss function
the loss function measures how far the prediction is from the true answer
a smaller loss means the model is making better predictions
mean squared error is a common loss function for regression problems
cross entropy is a common loss function for classification problems
what is gradient descent
gradient descent updates every weight in the direction that reduces the loss
the learning rate controls how big each weight update is
a learning rate that is too large makes the loss jump around
a learning rate that is too small makes the training very slow
what is backpropagation
backpropagation uses the chain rule to find the gradient of every weight
the error signal flows backward from the output layer to the input layer
the forward pass computes the prediction and the backward pass computes the gradients
after the backward pass the optimizer updates all the weights
what is an epoch
an epoch is one full pass over the whole training data
what is a batch
a batch is a small group of samples used for one weight update
mini batch training is faster and smoother than updating after every single sample
the optimizer adam adapts the learning rate for every weight
what is overfitting
overfitting happens when the model memorizes the training data
an overfitted model performs well on training data but badly on new data
validation data is used to check how the model performs on unseen data
early stopping ends the training when the validation loss stops improving
dropout randomly switches off some neurons during training to reduce overfitting
more training data usually helps the model generalize better
what is a recurrent neural network
a recurrent neural network reads a sequence one step at a time
at every step the recurrent neural network combines the new input with its previous hidden state
the hidden state acts as the memory of the network
the same weights are reused at every time step of the sequence
recurrent neural networks are used for text speech and time series data
a simple recurrent neural network forgets information from far back in the sequence
what is the vanishing gradient problem
the vanishing gradient problem makes it hard to learn long term patterns
gradients become smaller and smaller as they travel back through many time steps
the exploding gradient problem makes the weights grow too large
gradient clipping limits the size of the gradients to keep training stable
what is an lstm network
an lstm network adds a cell state to remember information for a long time
the forget gate decides how much of the old memory to keep
the input gate decides how much of the new information to store
the output gate decides how much of the memory to show as the hidden state
the cell state travels through time with only small changes
an lstm can remember important words from the beginning of a long sentence
a gru is a simpler version of an lstm with fewer gates
what is the difference between an rnn and an lstm
text must be converted into numbers before a neural network can read it
a tokenizer splits the text into words and gives every word a number
the tokenizer builds a vocabulary of all the distinct words in the text
padding makes every sequence the same length by adding zeros
what is an embedding layer
an embedding layer turns every word number into a vector of real numbers
words with similar meanings get similar vectors after training
the embedding layer has one learnable vector for every word in the vocabulary
what is a language model
a language model predicts the next word given the words that came before
the softmax layer gives a probability for every word in the vocabulary
the model picks the word with the highest probability as the next word
to generate a sentence the model predicts one word and adds it to the input
the new input is passed to the model again to predict the following word
a good language model needs a large amount of clean text
what is sentiment analysis
sentiment analysis decides whether a review is positive or negative
a sentiment model reads the review and outputs a probability between zero and one
a probability close to one means the review is positive
a probability close to zero means the review is negative
what is machine learning
machine learning is the science of learning patterns from data
supervised learning uses examples with the correct answers
unsupervised learning finds hidden patterns in data without any labels
classification predicts a category and regression predicts a number
the training set is used to learn and the test set is used to evaluate
a good model performs well on both the training set and the test set
python is the most popular language for machine learning
numpy is a python library for fast numerical computation
tensorflow is a library for building and training neural networks
keras is a simple interface for building models in tensorflow
a sequential model stacks layers one after another
to build a model in keras you add layers to a sequential model
the compile step chooses the loss function and the optimizer
the fit step trains the model on the training data
the predict step uses the trained model on new data
the model summary shows every layer and the number of parameters
a model with more layers and neurons has more parameters to learn
a larger model can learn more complex patterns but needs more data
a small model trains faster but may not learn complex patterns
a graphics card can train neural networks much faster than a normal processor
what is accuracy
the accuracy tells us what fraction of the predictions are correct
the loss should go down during training and the accuracy should go up
if the validation loss goes up while the training loss goes down the model is overfitting
a model that is too simple cannot learn the pattern in the data
what is deep learning
a deep network has many hidden layers between the input and the output
deep learning is machine learning with deep neural networks
every layer of a deep network learns a more abstract feature than the layer before
convolutional neural networks are used for images
recurrent neural networks and lstm networks are used for sequences
what is a transformer
a transformer reads all the words of a sentence at the same time
attention lets the model focus on the most important words
modern language models are built with transformers and trained on huge amounts of text
"""

# ---------------------------------------------------------------
# 1. Tokenize the text
# ---------------------------------------------------------------
tokenizer = Tokenizer()
tokenizer.fit_on_texts([faqs])

# FIX 1: Tokenizer numbers words from 1 (0 is reserved for padding),
# so we need one extra slot. This replaces the hardcoded 283.
vocab_size = len(tokenizer.word_index) + 1
print("Distinct words:", len(tokenizer.word_index), "-> vocab_size:", vocab_size)

# ---------------------------------------------------------------
# 2. Build n-gram sequences: "what is" -> "what is the" -> ...
# ---------------------------------------------------------------
input_sequences = []
for sentence in faqs.split('\n'):
    tokenized_sentence = tokenizer.texts_to_sequences([sentence])[0]
    for i in range(1, len(tokenized_sentence)):
        input_sequences.append(tokenized_sentence[:i + 1])

max_len = max([len(x) for x in input_sequences])
padded_input_sequences = pad_sequences(input_sequences, maxlen=max_len, padding='pre')

# Everything except the last word is the input, the last word is the target
X = padded_input_sequences[:, :-1]
y = padded_input_sequences[:, -1]
y = to_categorical(y, num_classes=vocab_size)

print("X shape:", X.shape, " y shape:", y.shape)

# ---------------------------------------------------------------
# 3. Model: Embedding -> LSTM -> LSTM -> softmax over all words
# ---------------------------------------------------------------
model = Sequential()
model.add(Input(shape=(max_len - 1,)))               # FIX 2: replaces input_length=56
model.add(Embedding(vocab_size, 100))                # FIX 1: vocab_size instead of 283
model.add(LSTM(150, return_sequences=True))
model.add(LSTM(150))
model.add(Dense(vocab_size, activation='softmax'))   # FIX 1: vocab_size instead of 283

model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
model.summary()

model.fit(X, y, epochs=100)

# ---------------------------------------------------------------
# 4. Generate text, one predicted word at a time
# ---------------------------------------------------------------
index_to_word = {index: word for word, index in tokenizer.word_index.items()}

text = "what is the fee"
print(text)

for i in range(10):
    token_text = tokenizer.texts_to_sequences([text])[0]
    # FIX 3: use max_len - 1 instead of the hardcoded 56
    padded_token_text = pad_sequences([token_text], maxlen=max_len - 1, padding='pre')
    pos = np.argmax(model.predict(padded_token_text, verbose=0))
    word = index_to_word.get(pos)
    if word is None:          # index 0 = padding, no word to add
        break
    text = text + " " + word
    print(text)
    time.sleep(1)