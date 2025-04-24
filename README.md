# GENERATIVE-TEXT-MODEL

*COMPANY*: CODTECH IT SOLUTIONS

*NAME*: CHINTAM VIJAYA LAKSHMI

*INTERN ID*: CT06WC94

*DOMAIN*: ARTIFICIAL INTELLIGENCE

*DURATION*: 6 WEEKS

*MENTOR*: NEELA SANTHOS

#CREATE A TEXT GENERATION MODEL USING GPT OR LSTM TO GENERATE COHERENT PARAGRAPHS ON SPECIFIC TOPICS.
#DELIVERABLE: A NOTEBOOK DEMONSTRATING GENERATED TEXT BASED ON USER PROMPTS.

#Text generation is a technique that involves the creation of human-like text using artificial intelligence 
#and machine learning algorithms. It enables computers to generate coherent and contextually relevant text 
#based on patterns and structures learned from existing textual data.

#Text generation is a fascinating application of deep learning in natural language processing (NLP). 
#It involves training a model on a given text dataset, which can then generate new, coherent sequences 
#of text based on the patterns it has learned.
#Define the Sample text that will be used to train the model 
#To train the model, the input text needs to be converted into a numerical format. 
#This is done using Keras’s Tokenizer, which converts the words in the text to sequences of integers. 
#Each unique word is assigned a specific index, and the tokenizer creates a mapping of words to indices.

#Tokenize the text
#fit_on_texts(): Creates the word-to-index mapping
#Stores the number of unique words in the given input text (plus one for padding).
#fit_on_texts used in conjunction with texts_to_sequences produces the one-hot encoding for a text
#Create sequences for text generation
#For example 'Morning Larks and night' will generate the below sequences
#"Morning Larks"
#"Morning Larks and"
#"Morning Larks and night"
#The above command split each sentence into sequences of increasing length, creating the n-gram sequences necessary for training.
#The sequences generated in the previous step are of varying lengths. Since the model requires all input sequences 
#to be of the same length, we pad them to a uniform size using pad_sequences(). 
#The padding is done at the beginning of the sequences to ensure that the sequences are aligned properly.
#input sequences are now padded with zeros
#Preparing the input and output for the model 
#Each sequence is split into input (X) and output (y). 
#The input consists of all words in the sequence except the last one
#the output is the last word in the sequence. 
#The output (y) is one-hot encoded to allow the model to predict the next word from a vocabulary of all possible words.
#Build LSTM model for text generation
#Embedding Layer - Converts the input word indices into dense vectors of fixed size.
#LSTM Layer - Processes the input sequences and learns the temporal relationships between words.
#Dense Layer - Outputs a probability distribution over the vocabulary using a softmax activation function, which predicts the next word in the sequence.
#Compile the model using the Adam optimizer and categorical crossentropy as the loss function. 
#Train the model for 100 epochs
#A trained model can be used to generate new text. 
#The function generate_text() takes a seed text and generates a specified number of new words by predicting
#the next word repeatedly, updating the seed text with the new predictions.
#Test your LSTM model. Pass a seed text and see how the model generate the text
