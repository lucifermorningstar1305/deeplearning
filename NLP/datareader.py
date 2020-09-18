
"""
@author Adityam Ghosh
"""

import numpy as np
import os
import re
import operator
from nltk.corpus import brown


class DataReader:

    def __init__(self):
        pass

    def readData(self, path="WikiQACorpus"):

        """
        ---------------------------------------------
        Description :
        Function to read data from file

        Input:
        path : file path

        Return:
        corpus : list of sentences 
        ----------------------------------------------
        """
        corpus = []
        with open(os.path.join(os.getcwd(),"WikiQACorpus/WikiQA-train.txt"), "r") as f:
            
            for line in f.readlines():

                corpus.append(line)

        return corpus

    def getVocabulary(self, corpus, vocab_limit=2000):

        """
        -----------------------------------------------------
        Description :
        Function to get the Vocabulary of a given corpus

        Input :
        corpus : list of sentences to deal with
        vocab_limit : the vocab size required

        Return:
        word2idx : the required sized vocabulary
        sentences : the required sized sentences as per the new vocabulary
        -------------------------------------------------------
        """
        word2idx = {}
        word_counter = {}
        sentences = []

        for sentence in corpus:
            sentence = sentence.lower()
            sentence = re.split(r"[^a-zA-Z0-9]|[\s\t\n]", sentence)
            sentence = list(filter(lambda x: x!='', sentence))

            for word in sentence:
                if word not in word_counter:
                    word_counter[word] = 0
                word_counter[word] += 1

        
        V = min(vocab_limit, len(word_counter))

        sorted_word_counter = sorted(word_counter.items(), key=operator.itemgetter(1), reverse=True)

        top_words = [w for w, wcount in sorted_word_counter[:V-1]] + ["UNK"]
        word2idx = {k:v for v, k in enumerate(top_words)}
        unknown = word2idx["UNK"]
        for sentence in corpus:
            sentence = sentence.lower()
            sentence = re.split(r"[^a-zA-Z0-9]|[\s\t\n]", sentence)
            sentence = list(filter(lambda x: x != "", sentence))
            if len(sentence) < 1:
                continue
            sentences.append([word2idx[word] if word in word2idx else unknown for word in sentence])

        return word2idx, sentences


    def readBrown(self):
        """
        ----------------------------------------
        Description :
        Function to read the Brown corpus from the nltk library and return it as a sentence

        Input:
        None

        Return:
        sentences -- a python list structure containing the sentences of the brown corpus
        ---------------------------------------
        """
        sentences = brown.sents()
        sentences = list(map(lambda x: ' '.join(x).lower(), sentences))
        return sentences





    def main(self, vocab_limit=2000, datatype="wiki"):

        """
        --------------------------------------------
        Description :
        A unified driver function for the class

        Input :
        vocab_limit : the required vocabulary size

        Return :
        Return the vocabulary
        ----------------------------------------------
        """
        corpus = ""
        if datatype == "wiki": 
            corpus = self.readData()
        elif datatype == "brown":
            corpus = self.readBrown()


        return self.getVocabulary(corpus, vocab_limit=vocab_limit)

