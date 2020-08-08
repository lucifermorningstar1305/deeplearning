from nltk.corpus import brown
import operator
import re

class Brown():

    def __init__(self):
        pass
    
    def remove_punctuation(self, sentence):

        """
        --------------------------------------------
        Description :
        Funcion to remove punctuation from a sentence

        Input :
        sentence : a string value

        Return :
        new_sentence : a string value 
        ---------------------------------------------
        
        """

        words = re.findall(r"""\w+[']+\w|\w+|[.]""", sentence)
        new_sentence = ' '.join(words)
        return new_sentence


    def get_vocab(self):

        """
        -----------------------------------------
        Description :
        Function to get the compelete Vocabulary list in a brown_corpus

        Return :
        word2idx : a dict-object having word:index pair
        sentences : a list of list object containing indices of every word as they are in sentence
        word2idx_count : a dict-object having word:count of word pair
        ------------------------------------------

        """

        brown_sentences = brown.sents()
        brown_sentences = list(map(lambda x: ' '.join(x).lower(), brown_sentences))

        new_brown_sentence = [self.remove_punctuation(sents) for sents in brown_sentences]

        word2idx = {"START" : 0, "END" : 1}
        idx2word = ["START", "END"]
        word2idx_count = {0 : float('inf'), 1 : float('inf')}
        idx = 2
        sentences = []

        for sent in new_brown_sentence:
            words = sent.split()
            for token in words:
                if token not in word2idx:
                    word2idx[token] = idx
                    idx += 1
                    idx2word.append(word2idx[token])
                
                word2idx_count[word2idx[token]] = word2idx_count.get(word2idx[token], 0) + 1
            sentences_by_idx = [word2idx[token] for token in words]
            sentences.append(sentences_by_idx)

        
        return word2idx, sentences, word2idx_count


    def get_limited_vocab(self, vocab_size = 200):
        """
        ----------------------------------------
        Description :
        Function to obtain limited brown corpus vocabulary

        Input:
        vocab_size : int value 

        Return :
        word2idx_small : a dict object
        sentences_small : a list of list structure
        ----------------------------------------

        """

        word2idx, sentences, word2idx_count = self.get_vocab()
        idx2word = [k for k,v in word2idx.items()]

        sorted_word2idx_count = sorted([(k, v) for k,v in word2idx_count.items()], key = operator.itemgetter(1), reverse = True)
        # print("Hello",sorted_word2idx_count)

        new_idx2idx_map = {}
        word2idx_small = {}
        sentences_small = []

        idx = 0

        for wordidx, count in sorted_word2idx_count[:vocab_size]:
            # print(wordidx)
            token = idx2word[wordidx]
            if token not in word2idx_small:
                word2idx_small[token] = idx
                new_idx2idx_map[wordidx] = idx
                idx += 1
        
        word2idx_small["UNKNOWN"] = idx
        unknown = idx


        for sentence in sentences:
            if len(sentence) > 1:
                new_sentence = [new_idx2idx_map[idx] if idx in new_idx2idx_map else unknown for idx in sentence]
                sentences_small.append(new_sentence)

        return word2idx_small, sentences_small

        
                
            




