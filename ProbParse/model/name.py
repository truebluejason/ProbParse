import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributions as dist
from typing import Sequence, Tuple

import const
from converter.converter import LetterSet
from neural_net.lstm import Encoder, Decoder


class NameVAE(nn.Module):
    def __init__(self, hidden_size: int, num_layers: int, test_mode: bool, temperature: float = 1.):
        super(NameVAE, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.test_mode = test_mode
        self.temperature = temperature
        self.permitted_set = LetterSet(allowed_letters=const.PERMITTED_LETTERS, allowed_length=const.MAX_NAME_LENGTH)
        self.fname_set = LetterSet(allowed_letters=const.LATENT_LETTERS, allowed_length=const.MAX_FIRST_NAME_LENGTH)
        self.mname_set = LetterSet(allowed_letters=const.LATENT_LETTERS, allowed_length=const.MAX_MIDDLE_NAME_LENGTH)
        self.lname_set = LetterSet(allowed_letters=const.LATENT_LETTERS, allowed_length=const.MAX_LAST_NAME_LENGTH)
        self.encoder = Encoder(input_size=self.permitted_set.n_words, hidden_size=hidden_size)
        self.format_decoder = Decoder(input_size=self.permitted_set.n_words, hidden_size=hidden_size, 
                                    output_size=6, max_length=1)
        self.fname_decoder = Decoder(input_size=self.fname_set.n_words, hidden_size=hidden_size, 
                                    output_size=self.fname_set.n_words, max_length=const.MAX_FIRST_NAME_LENGTH)
        self.mname_decoder = Decoder(input_size=self.mname_set.n_words, hidden_size=hidden_size, 
                                    output_size=self.mname_set.n_words, max_length=const.MAX_MIDDLE_NAME_LENGTH)
        self.lname_decoder = Decoder(input_size=self.lname_set.n_words, hidden_size=hidden_size, 
                                    output_size=self.lname_set.n_words, max_length=const.MAX_LAST_NAME_LENGTH)
    
    def forward(self, x: torch.tensor) -> Tuple[torch.tensor, torch.tensor, torch.tensor]:
        """
        Try to reconstruct the input tensor
        Input: One hot encoded strings <max name length, batch size, num permitted letters>
        Output: One hot encoded reconstructed strings, log probability of this batch, and KL divergence for each latents
        """
        # reconstruct x
        fname_info, mname_info, lname_info, format_info = self._encode(x)
        reconstructed = self._decode(fname_info[0], mname_info[0], lname_info[0], format_info[0])

        # compute the mean of this batch's log_prob
        batch_log_prob = self._get_categorical_log_prob(format_info[1], format_info[0])
        for info in [fname_info, mname_info, lname_info]:
            batch_log_prob += torch.sum(self._get_categorical_log_prob(info[1], info[0]), dim=0)
        total_log_prob = torch.mean(batch_log_prob)
        
        # compute the mean of this batch's KL Divergence for first, middle, last names and format
        # against the uniform categorical distribution
        kls = {}
        batch_size = x.shape[1]
        log_n = torch.log(torch.tensor(float(batch_size)).to(const.DEVICE))
        format_kl = torch.ones(batch_size).to(const.DEVICE) * log_n - self._get_categorical_entropy(format_info[1])
        fname_letter_kl = torch.ones(self.fname_set.n_length, batch_size).to(const.DEVICE) * log_n - self._get_categorical_entropy(fname_info[1])
        mname_letter_kl = torch.ones(self.mname_set.n_length, batch_size).to(const.DEVICE) * log_n - self._get_categorical_entropy(mname_info[1])
        lname_letter_kl = torch.ones(self.lname_set.n_length, batch_size).to(const.DEVICE) * log_n - self._get_categorical_entropy(lname_info[1])
        kls['format'] = torch.mean(format_kl)
        kls['fname'] = torch.mean(torch.sum(fname_letter_kl, dim=0))
        kls['mname'] = torch.mean(torch.sum(mname_letter_kl, dim=0))
        kls['lname'] = torch.mean(torch.sum(lname_letter_kl, dim=0))

        return reconstructed, total_log_prob, kls
    
    def predict(self, x: torch.tensor) -> Sequence[Tuple[str, str, str, int]]:
        """
        Input: One hot encoded strings <max name length, batch size, num permitted letters>
        Output: List of first, middle, last names and format
        """
        fname_info, mname_info, lname_info, format_info = self._encode(x)

        batch_shape = x.shape[1]
        fnames = self.fname_set.to_words(fname_info[0])
        mnames = self.mname_set.to_words(mname_info[0])
        lnames = self.lname_set.to_words(lname_info[0])
        format_ids = torch.argmax(format_info[0], dim=2).squeeze().numpy().tolist()
        
        result = []
        for index in range(batch_shape):
            result.append((fnames[index], mnames[index], lnames[index], format_ids[index]))
        return result
    
    def _encode(self, x: torch.tensor) -> Sequence[Tuple[torch.tensor, torch.tensor]]:
        """
        Use sequence to sequence LSTM to obtain distribution over latent variables
        Input: One hot encoded strings <max name length, batch size, num permitted letters>
        Output: One hot encoded samples and categorical parameters of first, middle, last names and name format
        """
        batch_size = x.shape[1]
        encoder_hidden_states = torch.zeros(const.MAX_NAME_LENGTH, batch_size, self.num_layers*self.hidden_size*2).to(const.DEVICE)
        curr_hidden_state = self.encoder.init_hidden(batch_size=batch_size)
        for i in range(const.MAX_NAME_LENGTH):
            _, curr_hidden_state = self.encoder.forward(x[i].unsqueeze(0), curr_hidden_state)
            encoder_hidden_states[i] = torch.cat((curr_hidden_state[0], curr_hidden_state[1]), dim=2).squeeze(0)
        
        mask = self.permitted_set.padding_tensor(x)
        format_probs = self._get_format_probs(curr_hidden_state, encoder_hidden_states, mask, batch_size)
        fname_probs = self._get_name_character_probs(self.fname_decoder, self.fname_set, 
                                                     curr_hidden_state, encoder_hidden_states, mask, batch_size)
        mname_probs = self._get_name_character_probs(self.mname_decoder, self.mname_set, 
                                                     curr_hidden_state, encoder_hidden_states, mask, batch_size)
        lname_probs = self._get_name_character_probs(self.lname_decoder, self.lname_set, 
                                                     curr_hidden_state, encoder_hidden_states, mask, batch_size)
        
        format_sample = self._sample_categorical(format_probs)
        fname_sample = self._sample_categorical(fname_probs)
        mname_sample = self._sample_categorical(fname_probs)
        lname_sample = self._sample_categorical(fname_probs)
        return (fname_sample,fname_probs), (mname_sample,mname_probs), (lname_sample,lname_probs), (format_sample,format_probs)

    def _decode(self, fname_sample: torch.tensor, mname_sample: torch.tensor, lname_sample: torch.tensor, format_sample: torch.tensor) -> torch.tensor:
        """
        Based on the format, deterministically join first, middle, last names into a reconstructed tensor
        then smooth through softmax
        """
        batch_shape = format_sample.shape[1]
        names = [None] * batch_shape
        fnames = self.fname_set.to_words(fname_sample)
        mnames = self.mname_set.to_words(mname_sample)
        lnames = self.lname_set.to_words(lname_sample)

        format_ids = torch.argmax(format_sample, dim=2)
        for index in range(batch_shape):
            format_id = format_ids[0,index]
            names[index] = self._format_name(fnames[index], mnames[index], lnames[index], format_id)
        tensor = self.permitted_set.to_one_hot_tensor(names)
        return F.softmax(tensor/self.temperature, dim=2)

    def _format_name(self, fname, mname, lname, format_id) -> str:
        if format_id == 0:
            return f"{fname} {lname}"
        elif format_id == 1:
            return f"{fname} {mname} {lname}"
        elif format_id == 2:
            return f"{lname}, {fname}"
        elif format_id == 3:
            return f"{lname}, {fname} {mname}"
        elif format_id == 4:
            return f"{fname} {mname}. {lname}"
        elif format_id == 5:
            return f"{lname}, {fname} {mname}."
    
    def _get_format_probs(self, curr_hidden_state, encoder_hidden_states, mask, batch_size) -> torch.tensor:
        initial_tensor = self.permitted_set.initial_tensor(batch_size)
        probs, _ = self.format_decoder.forward(initial_tensor, curr_hidden_state, encoder_hidden_states, mask)
        return probs
    
    def _get_name_character_probs(self, decoder, letter_set, curr_hidden_state, encoder_hidden_states, mask, batch_size) -> torch.tensor:
        name_probs = torch.zeros(letter_set.n_length, batch_size, letter_set.n_words).to(const.DEVICE)
        input_tensor = letter_set.initial_tensor(batch_size)
        for index in range(letter_set.n_length):
            character_prob, curr_hidden_state = decoder.forward(input_tensor, curr_hidden_state, encoder_hidden_states, mask)
            name_probs[index] = character_prob.squeeze(0)
            input_tensor = character_prob
        return name_probs
    
    def _get_categorical_log_prob(self, probs, sample):
        if self.test_mode:
            return dist.OneHotCategorical(probs=probs).log_prob(sample)
        else:
            # RelaxedOneHotCategorical can't compute log_prob - this should be similar
            return dist.OneHotCategorical(probs=probs).log_prob(sample)
    
    def _get_categorical_entropy(self, probs):
        if self.test_mode:
            return dist.OneHotCategorical(probs=probs).entropy()
        else:
            # RelaxedOneHotCategorical can't compute entropy - this should be similar
            return dist.OneHotCategorical(probs=probs).entropy()

    def _sample_categorical(self, probs):
        if self.test_mode:
            return dist.OneHotCategorical(probs=probs).sample()
        else:
            return dist.RelaxedOneHotCategorical(temperature=self.temperature, probs=probs).sample()
