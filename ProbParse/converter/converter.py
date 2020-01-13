import string
import torch

import const

class LetterSet:
    def __init__(self, allowed_letters, allowed_length):
        self.word2index = {}
        self.index2word = {}
        self.n_length = allowed_length
        self.n_words = 0
        for char in allowed_letters:
            self._add_letter(char)
    
    def to_one_hot_tensor(self, words):
        tensor = torch.zeros(self.n_length, len(words), self.n_words).to(const.DEVICE)
        padded_words = list(map(lambda s: self._pad_string(s, self.n_length), words))
        for i_name, name in enumerate(padded_words):
            for i_char, letter in enumerate(name):
                tensor[i_char][i_name][self.word2index[letter]] = 1.
        return tensor
    
    def to_words(self, one_hot_tensor):
        words = [''] * one_hot_tensor.shape[1]
        for i_name in range(one_hot_tensor.shape[1]):
            for i_char in range(one_hot_tensor.shape[0]):
                nonzero_index = torch.argmax(one_hot_tensor[i_char,i_name]).item()
                if self.index2word[nonzero_index] == const.EOS_TOKEN:
                    break
                words[i_name] += self.index2word[nonzero_index]
        return words
    
    def initial_tensor(self, batch_size):
        tensor = torch.zeros(1, batch_size, self.n_words).to(const.DEVICE)
        tensor[0,:,self.word2index[const.EOS_TOKEN]] = 1.
        return tensor
    
    def padding_tensor(self, one_hot_tensor):
        tensor = torch.zeros(1,one_hot_tensor.shape[1], self.n_length, dtype=torch.bool).to(const.DEVICE)
        for i_seq in range(one_hot_tensor.shape[0]):
            for i_batch in range(one_hot_tensor.shape[1]):
                if one_hot_tensor[i_seq,i_batch,self.word2index[const.PAD_TOKEN]] == 1.:
                    tensor[0,i_batch,i_seq] = 1
        return tensor

    def _add_letter(self, word):
        if word not in self.word2index:
            self.word2index[word] = self.n_words
            self.index2word[self.n_words] = word
            self.n_words += 1

    def _pad_string(self, original: str, desired_len: int, pad_character: str = const.PAD_TOKEN):
        return original + (pad_character * (desired_len - len(original)))
