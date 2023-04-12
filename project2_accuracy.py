# Project2 for EN.520.666 Information Extraction

# 2021 Ruizhe Huang
# 2022 Zili Huang

import numpy as np
import matplotlib.pyplot as plt
import string

import os
from HMM import HMM
from collections import Counter
from collections import defaultdict
from itertools import chain
from tqdm import tqdm
from scipy.special import logsumexp
import pickle
import argparse

NOISE = "<noise>"
data_dir = "./data"

def read_file_line_by_line(file_name, func=lambda x: x, skip_header=True):
    print("reading file: %s" % file_name)
    res = list()
    with open(file_name, "r") as fin:
        if skip_header:
            fin.readline()  # skip the header
        for line in fin:
            if len(line.strip()) == 0:
                continue
            fields = func(line.strip())
            res.append(fields)
    print("%d lines, done" % len(res))
    return res


class Word_Recognizer:

    def __init__(self, args):
        # read labels
        self.lblnames = read_file_line_by_line(os.path.join(data_dir, "clsp.lblnames"))

        # read training data
        self.trnlbls = read_file_line_by_line(os.path.join(data_dir, "clsp.trnlbls"), func=lambda x: x.split())
        self.endpts = read_file_line_by_line(os.path.join(data_dir, "clsp.endpts"), func=lambda x: list(map(int, x.split())))
        self.trnscr = read_file_line_by_line(os.path.join(data_dir, "clsp.trnscr"))

        # read dev data
        self.devlbls = read_file_line_by_line(os.path.join(data_dir, "clsp.devlbls"), func=lambda x: x.split())
        self.train_words = set(self.trnscr)

        assert len(self.trnlbls) == len(self.endpts)
        assert len(self.trnlbls) == len(self.trnscr)

        # 23 letters + noise
        self.letters = list(string.ascii_lowercase)
        for c in ['k', 'q', 'z']:
            self.letters.remove(c)
        self.noise_id = len(self.letters)
        self.letters.append(NOISE)
        self.letter2id = dict({c: i for i, c in enumerate(self.letters)})
        self.id2letter = dict({i: c for c, i in self.letter2id.items()})

        # 256 quantized feature-vector labels
        self.label2id = dict({lbl: i for i, lbl in enumerate(self.lblnames)})
        self.id2label = dict({i: lbl for lbl, i in self.label2id.items()})

        # convert file contents to integer ids
        self.trnlbls = [[self.label2id[lbl] for lbl in line] for line in self.trnlbls]
        self.devlbls = [[self.label2id[lbl] for lbl in line] for line in self.devlbls]
        self.trnscr = [[self.letter2id[c] for c in word] for word in self.trnscr]

        # get label frequencies
        lbl_freq = self.get_unigram(self.trnlbls, len(self.lblnames), smooth=0)
        lbl_freq_noise = self.get_unigram(self.trnlbls, len(self.lblnames), smooth=1, endpts=self.endpts)

        # get hmms for each letter
        self.letter_id2hmm = self.init_letter_hmm(lbl_freq, lbl_freq_noise, self.id2letter)

        self.find_N = args.find_N
        self.num_epochs = args.num_epochs

    def get_unigram(self, trnlbls, nlabels, smooth=0, endpts=None):
        # Compute "unigram" frequency of the training labels
        # Return freq(np array): the "unigram" frequency of the training labels
        #  Initialize all the emission distributions using the “unigram” frequency of the training labels in clsp.trnlbls.
        
        if endpts is None:
            freq = Counter(chain.from_iterable(trnlbls))
            freq = np.array([freq[i] + smooth for i in range(nlabels)])
        else:
            # only consider strings from 0:st+1 and ed:
            trnlbls = [lbls[0:st+1] + lbls[ed:] for lbls, (st, ed) in zip(trnlbls, endpts)]
            freq = Counter(chain.from_iterable(trnlbls))
            freq = np.array([freq[i] + smooth for i in range(nlabels)])

        freq = freq / freq.sum()
        return freq

    def init_letter_hmm(self, lbl_freq, lbl_freq_noise, id2letter):
        # Initialize the HMM for each letter
        # Return letter_id2hmm(dict): the key is the letter_id and the value is the corresponding HMM
        
        letter_id2hmm = {}
        for letter_id in range(len(id2letter)):
            if letter_id == self.noise_id:
                # HMM for silence and non-speech sounds
                letter_id2hmm[letter_id] = HMM(num_states=5, num_outputs=len(self.lblnames))
                letter_id2hmm[letter_id].init_transition_probs(np.asarray([[0.25, 0.25, 0.25, 0.25, 0.0], 
                                                                           [0.0, 0.25, 0.25, 0.25, 0.25], 
                                                                           [0.0, 0.25, 0.25, 0.25, 0.25], 
                                                                           [0.0, 0.25, 0.25, 0.25, 0.25], 
                                                                           [0.0, 0.0, 0.0, 0.0, 0.75]], dtype=np.float64))
                
                emission_probs = np.zeros((len(self.lblnames), 5, 5))
                # arc with non-zero transition probability
                non_zero_arcs = np.where(letter_id2hmm[letter_id].transitions > 0)
                for i, j in zip(non_zero_arcs[0], non_zero_arcs[1]):
                    emission_probs[:, i, j] = lbl_freq_noise

                letter_id2hmm[letter_id].init_emission_probs(np.asarray(emission_probs, dtype=np.float64))

                letter_id2hmm[letter_id].exiting_prob = 0.25
            else:
                # HMM for letters
                letter_id2hmm[letter_id] = HMM(num_states=3, num_outputs=len(self.lblnames))
                letter_id2hmm[letter_id].init_transition_probs(np.asarray([[0.8, 0.2, 0.0], 
                                                                           [0.0, 0.8, 0.2], 
                                                                           [0.0, 0.0, 0.8]], dtype=np.float64))
                
                emission_probs = np.zeros((len(self.lblnames), 3, 3))
                # arc with non-zero transition probability
                non_zero_arcs = np.where(letter_id2hmm[letter_id].transitions > 0)
                for i, j in zip(non_zero_arcs[0], non_zero_arcs[1]):
                    emission_probs[:, i, j] = lbl_freq

                letter_id2hmm[letter_id].init_emission_probs(np.asarray(emission_probs, dtype=np.float64))

                letter_id2hmm[letter_id].exiting_prob = 0.2

        return letter_id2hmm

    def id2word(self, w):
        # w should be a list of char ids
        return ''.join(map((lambda c: self.id2letter[c]), w))

    def get_word_model(self, scr):
        # Construct the word HMM based on self.letter_id2hmm by 
        # Return h(HMM object): the word HMM for the word scr 
        
        transition_matrix = np.zeros((5*2 + len(scr)*3, 5*2 + len(scr)*3))
        emission_matrix = np.zeros((len(self.lblnames), 5*2 + len(scr)*3, 5*2 + len(scr)*3))
        null_arcs = defaultdict(dict)
        h = HMM(num_states=5*2 + len(scr)*3, num_outputs=len(self.lblnames))
               
        # add SIL transition to the left upper and right lower corner of transition_matrix
        transition_matrix[0:5,0:5] = self.letter_id2hmm[self.noise_id].transitions
        transition_matrix[-5:,-5:] = self.letter_id2hmm[self.noise_id].transitions

        # add SIL emission to the left upper and right lower corner of emission_matrix
        emission_matrix[:,0:5,0:5] = self.letter_id2hmm[self.noise_id].emissions
        emission_matrix[:,-5:,-5:] = self.letter_id2hmm[self.noise_id].emissions

        # add null arc of existing SIL HMM
        null_arcs[4][5] = self.letter_id2hmm[self.noise_id].exiting_prob

        # add letter HMM transitions and emissions
        for idx, letter_id in enumerate(scr):
            transition_matrix[5+idx*3:5+(idx+1)*3, 5+idx*3:5+(idx+1)*3] = self.letter_id2hmm[letter_id].transitions
            emission_matrix[:, 5+idx*3:5+(idx+1)*3, 5+idx*3:5+(idx+1)*3] = self.letter_id2hmm[letter_id].emissions
            null_arcs[4+(idx+1)*3][5+(idx+1)*3] = self.letter_id2hmm[letter_id].exiting_prob
        
        h.init_transition_probs(np.asarray(transition_matrix, dtype=np.float64))
        h.init_emission_probs(np.asarray(emission_matrix, dtype=np.float64))
        h.init_null_arcs(null_arcs)

        return h

    def update_letter_counters(self, scr, lbls, word_hmm):
        # Update self.letter_id2hmm based on the counts from word_hmm

        init_prob=np.asarray([1] + [0] * (word_hmm.num_states - 1), dtype=np.float64)
        # Reset the counters each forward-backward pass
        alphas_, betas_, Q = word_hmm.forward_backward(lbls, init_prob, update_params=False)
    
        self.letter_id2hmm[self.noise_id].output_arc_counts += word_hmm.output_arc_counts[:, 0:5, 0:5]
        self.letter_id2hmm[self.noise_id].output_arc_counts += word_hmm.output_arc_counts[:, -5:, -5:]

        for idx, letter_id in enumerate(scr):
            # update non-null arc counter for letter HMM
            self.letter_id2hmm[letter_id].output_arc_counts += word_hmm.output_arc_counts[:, 5+idx*3:5+(idx+1)*3, 5+idx*3:5+(idx+1)*3]

    def update_params(self):
        # update the parameters of the 23 letter HMMs and 1 SIL HMM after all 798 forward-backward passes are completed, 
        
        # update SIL HMM
        self.letter_id2hmm[self.noise_id].update_params()

        # update letter HMMs
        for letter_id in self.letter_id2hmm.keys():
            if letter_id != self.noise_id:
                self.letter_id2hmm[letter_id].update_params()

    def data_split(self, train_ratio=0.8):
        """
        Split the data into training and validation sets.
        Balance the distribution of word-tokens between the two
        """
        np.random.seed(0)
        # split the data into training and validation sets
        trnscr, trnlbls, valscr, vallbls = [], [], [], []
        for scr, lbls in zip(self.trnscr, self.trnlbls):
            if np.random.rand() < train_ratio:
                trnscr.append(scr)
                trnlbls.append(lbls)
            else:
                valscr.append(scr)
                vallbls.append(lbls)

        print(f"Kept set size: {len(trnscr)}")
        print(f"Held-out set size: {len(valscr)}")

        return trnscr, trnlbls, valscr, vallbls
        
    
    def train(self):
        if self.find_N:
            trnscr, trnlbls, valscr, vallbls = self.data_split(train_ratio=0.8)

            # sort trnlbls, endpts and trnscr such that the same word appear next to each other
            trnlbls_sorted = []
            trnscr_sorted = []
            for scr, lbls in sorted(zip(trnscr, trnlbls)):
                trnlbls_sorted.append(lbls)
                trnscr_sorted.append(scr)
        else:
            # sort trnlbls, endpts and trnscr such that the same word appear next to each other
            trnlbls_sorted = []
            trnscr_sorted = []
            for scr, lbls in sorted(zip(self.trnscr, self.trnlbls)):
                trnlbls_sorted.append(lbls)
                trnscr_sorted.append(scr)

        log_likelihood_list = []
        per_frame_log_likelihood_list = []

        N = 0 # iteration
        prev_acc = 0

        # training for this many epochs
        for i_epoch in range(self.num_epochs):
            print("---- echo: %d ----" % i_epoch)
            log_likelihood = 0
            num_frames = 0

            # reset the counters of the 23 letter HMMs and 1 SIL HMM
            for letter_id in self.letter_id2hmm.keys():
                self.letter_id2hmm[letter_id].reset_counters()

            for scr, lbls in zip(trnscr_sorted, trnlbls_sorted):
                # create a word HMM for each word
                word_hmm = self.get_word_model(scr)
                # update the counters of the 23 letter HMMs and 1 SIL HMM
                self.update_letter_counters(scr, lbls, word_hmm)
                # compute the log likelihood of the word HMM
                log_likelihood += word_hmm.compute_log_likelihood(data = lbls, 
                                                                  init_prob = np.asarray([1] + [0] * (word_hmm.num_states - 1), dtype=np.float64), 
                                                                  init_beta = np.ones(word_hmm.num_states))
                num_frames += len(lbls)
            
            log_likelihood_list.append(log_likelihood)
            per_frame_log_likelihood_list.append(log_likelihood / num_frames)
            print("log_likelihood =", log_likelihood, "per_frame_log_likelihood =", log_likelihood / num_frames)
            
            self.update_params()
            if self.find_N:
                acc = self.test_val_acc(valscr, vallbls)
                if acc > prev_acc:
                    prev_acc = acc
                    N += 1
                else:
                    break
        if self.find_N:
            print(f"Best validation accuracy: {prev_acc} at epoch {N} out of {self.num_epochs} epochs")
        else:       
            self.test()
            # plot the log likelihood over epochs
            plt.figure()
            plt.plot(log_likelihood_list)
            plt.xlabel("Epochs")
            plt.ylabel("Log Likelihood")
            plt.title("Log Likelihood over Epochs")
            plt.savefig("log_likelihood_N.png", dpi=300)

            # plot the per frame log likelihood over epochs
            plt.figure()
            plt.plot(per_frame_log_likelihood_list)
            plt.xlabel("Epochs")
            plt.ylabel("Per Frame Log Likelihood")
            plt.title("Per Frame Log Likelihood over Epochs")
            plt.savefig("per_frame_log_likelihood_N.png", dpi=300)


    def test_val_acc(self, valscr, vallbls):
        # test the accuracy of the model on the validation vallbls
        id2words = dict({i: w for i, w in enumerate(self.train_words)})
        words2id = dict({w: i for i, w in id2words.items()})

        word_likelihoods = np.zeros((len(words2id), len(vallbls)))

        for vallbl_id, vallbl in enumerate(vallbls):
            for word, word_id in words2id.items():
                dev_scr = [self.letter2id[c] for c in word]
                word_hmm = self.get_word_model(dev_scr)

                init_prob = np.asarray([1] + [0] * (word_hmm.num_states - 1), dtype=np.float64)
                init_beta = np.ones(word_hmm.num_states)

                word_likelihoods[word_id, vallbl_id] = word_hmm.compute_log_likelihood(data = vallbl, 
                                                                                       init_prob = init_prob, 
                                                                                       init_beta = init_beta)
            
        # find the most likely word for each dev sample
        result = word_likelihoods.argmax(axis=0)
        result = [id2words[res] for res in result]
        result_scr = [[self.letter2id[c] for c in word] for word in result]

        # compute the accuracy
        acc = 0
        for i in range(len(result_scr)):
            if result_scr[i] == valscr[i]:
                acc += 1
        acc /= len(result_scr)
        print("Hold-out accuracy: ", acc)
        return acc

    def test(self):
        # Compute the word likelihood for each dev samples 
        id2words = dict({i: w for i, w in enumerate(self.train_words)})
        words2id = dict({w: i for i, w in id2words.items()})

        word_likelihoods = np.zeros((len(words2id), len(self.devlbls)))

        for devlbl_id, devlbl in enumerate(self.devlbls):
            for word, word_id in words2id.items():
                dev_scr = [self.letter2id[c] for c in word]
                word_hmm = self.get_word_model(dev_scr)

                init_prob = np.asarray([1] + [0] * (word_hmm.num_states - 1), dtype=np.float64)
                init_beta = np.ones(word_hmm.num_states)

                word_likelihoods[word_id, devlbl_id] = word_hmm.compute_log_likelihood(data = devlbl, 
                                                                                      init_prob = init_prob, 
                                                                                      init_beta = init_beta)

        result = word_likelihoods.argmax(axis=0)
        result = [id2words[res] for res in result]

        # divide the forward-probability of the most likely word by the sum of the forward probabilities of all the 48 words
        confidence = np.exp(word_likelihoods.max(axis=0) - logsumexp(word_likelihoods, axis=0))
        confidence[np.isnan(confidence)] = 1.0

        # the most likely word and its confidence
        print(result)
        print(confidence)

    def save(self, i_epoch):
        fn = os.path.join(data_dir, "%d.mdl.pkl" % i_epoch)
        print("Saved to:", fn)
        for letter_id, hmm in self.letter_id2hmm.items():
            hmm.output_arc_counts = None
            hmm.output_arc_counts_null = None
        pickle.dump(self.letter_id2hmm, open(fn, "wb"))

    def load(self, i_epoch):
        return pickle.load(open(os.path.join(data_dir, "%d.mdl.pkl" % i_epoch), "rb"))

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--find_N",  action="store_true")
    parser.add_argument("--num_epochs", type=int, default=10)

    wr = Word_Recognizer(parser.parse_args())

    wr.train()

if __name__ == '__main__':
    main()
