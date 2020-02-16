import re
import string
import numpy as np
class CBOWmode:

    def __init__(self):
        pass

    def oneHot(self, idx, vocab_size):
        one_hot = np.zeros(vocab_size)
        one_hot[idx] = 1
        return one_hot

    def acceptString(self, data, WINDOW_SIZE=5):
        
        CONTEXT_SIZE = WINDOW_SIZE // 2
        data = data.lower()
        sentences = data.split('.') # Split the data into sentences
        sentences = [sent.translate(str.maketrans('','',string.punctuation)) for sent in sentences] ## Remove punctuation from a sentence
        wordlist = [word.split('.')[0] for sent in sentences for word in sent.split()] ## Getting the words in the input text
        """
        A vocabulary is a set of unique words in the text
        
        """
        vocabulary = set(wordlist)
        vocab_size = len(vocabulary)

        """
        1) Get word2idx 
        2) Get idx2word

        """
        word2idx = {word:idx for idx,word in enumerate(vocabulary)}
        idx2word = {idx:word for idx,word in enumerate(vocabulary)}
        _data = []
        for i in range(CONTEXT_SIZE, len(wordlist)-CONTEXT_SIZE):
            context = [wordlist[i-2],wordlist[i-1],wordlist[i+1],wordlist[i+2]]
            target = wordlist[i]
            _data.append((context, target))
        print(_data)
        Xtrain, ytrain = [], []
        for inputs, targets in _data:

            _t = []
            for _input in inputs:
                _t.append(word2idx[_input])
            Xtrain.append(_t)
            ytrain.append(word2idx[targets])

        return Xtrain, ytrain, word2idx, idx2word, vocab_size


