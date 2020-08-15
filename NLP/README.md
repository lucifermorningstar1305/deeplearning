# Projects in NLP

* **Glove_Word_Analogies** : The purpose of this notebook is to showcase how we can derive word analogies such as :

  * *king* - *man* = *queen* - *woman*
  * *India* - *Indians* = *America* - *Americans* 

  using the Glove Vectors. I have used a pre-trained GloVe vector model for the notebook , it's [download link](http://nlp.stanford.edu/data/glove.6B.zip)

* **BOW_classifier** : The purpose of this notebook is to evaluate the accuracies of a Glove vector model and a Word2vec vector model, though I haven't use the Word2Vec as it required GBs to download but you can do that as a challenge. 

* **BigramModel** : The purpose of this notebook is to demonstrate the Bigram Model or rather the Markov Model.

* **NeuralBigramModel** : The purpose of this notebook is to demonstrate the BigramModel built using just a single neuron.

* **NeuralNetworkBigramModel** : The purpose of this notebook is to demonstrate the BigramModel built using 2-layer neural network

* **ImprovisedNeuralNetworkBigram** : The purpose of this notebook is to demonstrate the NeuralNetworkBigram Model much more memory efficient.

# Understanding the BOW_classifier transform function

The idea of this function is to transform a word into it's vector form using any vector model (such as Glove, Word2Vec etc.)

Now basically what this function is doing is that it takes in input a list of sentence and loops through every sentence.

**The goal** : To convert each word of a sentence into it's vector form. So for example , if we have a sentence like this :

```python
sentence = "I Like Dogs"
#Then we have to split the sentence into words using the split() function
sentence = sentence.lower().split()
#So the variable sentence becomes like this 
#sentence = ["i", "like", "dogs"]

for word in sentence:
    if word in word2vec:
        vector = word2vec[vector]
        vectors.append(vector)
if len(vectors) > 0:
    vectors = np.array(vectors)
    X[n] = np.mean(vectors, axis=0) # This will calculate the mean of all the vector, I'll list the formula
    n += 1
```

Now as you can see I wrote a line like this :

```python
X[n] = np.mean(vectors, axis=0)
```

Why did I write this line ? Well because after we split the sentence into it's word , we get a list like this 

```python
words = ["i", "like", "eggs"]
```

Now we need to calculate the vector of each word

```
words = [vector("i"), vector("like"), vector("eggs")]
```

The Bag of Words formula is as follows:

```python
sum(vec(w) for w in S) / len(sentence)
```



Thus the above formula is used in the line 

```python
X[n] = np.mean(vectors, axis=0)
```

where X is a feature vector for the sentence "I like eggs"





# Bigram Model (aka Markov Model)

Formula for Bigram Model is :

```
p(B|A) = count(A -> B) / count(A)
```

The formula has one problem : 

If our model has a small vocabulary and if we input a custom sentence such as "The quick brown fox jumps over the dog"  then the output probability will be 0 if there is no word such as "turtle". Hence to counter such behaviour , *Add-One smoothning* is used 

```
p(B|A) = (count(A -> B) + 1) / (count(A) + vocab_size)
```



