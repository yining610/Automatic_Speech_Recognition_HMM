# Project1 for EN.520.666 Information Extraction

# 2021 Matthew Ost
# 2021 Ruizhe Huang
# 2022 Zili Huang

import numpy as np
import string
from numpy import log, add, exp
from scipy.special import logsumexp
# import time
import matplotlib.pyplot as plt

EPS = 1e-6

def load_data(fname):
    alphabet_string = string.ascii_lowercase
    char_list = list(alphabet_string)
    print(char_list)
    char_list.append(' ')
    with open(fname, 'r') as fh:
        content = fh.readline()
    content = content.strip('\n')
    data = []
    for c in content:
        assert c in char_list
        data.append(char_list.index(c))
    return np.array(data) 

def get_init_prob_2states():
    # Define initial transition probability and emission probability
    # for 2 states HMM
    # Transition probability of size [num_states x num_states]
    T_prob = np.array([[0.49, 0.51], [0.51, 0.49]])   # Q1, Q3
    # T_prob = np.array([[0.5, 0.5], [0.5, 0.5]])     # Q1
    # T_prob = np.array([[0.24, 0.24, 0.26, 0.26], [0.24, 0.24, 0.26, 0.26], [0.26, 0.26, 0.24, 0.24], [0.26, 0.26, 0.24, 0.24]]) #Q2
    # T_prob = np.array([[0.25]*4, [0.25]*4, [0.25]*4, [0.25]*4]) # Q2

    # emission probability of size [num_states x num_outputs]: 2x27
    E_prob = np.array([[0.037]*13 + [0.0371]*13 + [0.0367]*1, [0.0371]*13 + [0.037]*13 + [0.0367]*1]) # Q1
    # E_prob = np.array([[1/27]*27, [1/27]*27]) # Q1
    # E_prob = np.array([[0.037]*13 + [0.0371]*13 + [0.0367]*1, [0.0371]*13 + [0.037]*13 + [0.0367]*1, [0.037]*13 + [0.0371]*13 + [0.0367]*1, [0.0371]*13 + [0.037]*13 + [0.0367]*1]) # Q2
    # E_prob = np.array([[1/27]*27, [1/27]*27, [1/27]*27, [1/27]*27]) # Q2

    # Q3 
    # Compute the relative frequency q(y) of the letters from the entire text A
    # train_data = load_data('textA.txt')
    # num_outputs = 27
    # q = np.zeros((num_outputs))
    # for i in range(len(train_data)):
    #     q[train_data[i]] += 1
    # q = q/len(train_data)
    # # Generate a vector of random numbers r(y), compute the average r
    # r = np.random.rand(num_outputs)
    # r_bar = np.sum(r) / num_outputs
    # # create a zero-mean perturbation vector
    # delta = r - r_bar

    # l = 0.1
    # E_prob = np.array([q-l*delta, q+l*delta])
    # # dealing with negative values 
    # E_prob = np.where(E_prob<0, 0, E_prob)

    return T_prob, E_prob

class HMM:

    def __init__(self, num_states, num_outputs):
        # Args:
        #     num_states (int): number of HMM states
        #     num_outputs (int): number of output symbols            

        self.states = np.arange(num_states)  # just use all zero-based index
        self.outputs = np.arange(num_outputs)
        self.num_states = num_states
        self.num_outputs = num_outputs

        # Probability matrices
        self.transitions = None
        self.emissions = None

    def initialize(self, T_prob, E_prob):
        # Initialize HMM with transition probability T_prob and emission probability
        # E_prob

        # Args:
        #     T_prob (numpy.ndarray): [num_states x num_states] numpy array.
        #     T_prob[i, j] is the transition probability from state i to state j.
        #     E_prob (numpy.ndarray): [num_states x num_outputs] numpy array.
        #     E_prob[i, j] is the emission probability of state i to output jth symbol. 
        self.transitions = T_prob
        self.emissions = E_prob
        self._assert_transition_probs()
        self._assert_emission_probs()

    def _assert_emission_probs(self):
        for s in self.states:
            assert self.emissions[s].sum() - 1 < EPS

    def _assert_transition_probs(self):
        for s in self.states:
            assert self.transitions[s].sum() - 1 < EPS
            assert self.transitions[:, s].sum() - 1 < EPS

    # def printAB(self) -> None:
    #     """Print the A and B matrices in a more human-readable format (tab-separated)."""
    #     print("Transition matrix A:")
    #     col_headers = [""] + [str(self.states[t]) for t in range(self.transitions.shape[1])]
    #     print("\t".join(col_headers))
    #     for s in range(self.transitions.shape[0]):   # rows
    #         row = [str(self.states[s])] + [f"{self.transitions[s,t]:.5f}" for t in range(self.transitions.shape[1])]
    #         print("\t".join(row))
    #     print("\nEmission matrix B:")        
    #     col_headers = [""] + [str(self.outputs[w]) for w in range(self.emissions.shape[1])]
    #     print("\t".join(col_headers))
    #     for t in range(self.transitions.shape[0]):   # rows
    #         row = [str(self.states[t])] + [f"{self.emissions[t,w]:.5f}" for w in range(self.emissions.shape[1])]
    #         print("\t".join(row))
    #     print("\n")

    def Baum_Welch(self, max_iter, train_data, test_data):
        # The Baum Welch algorithm to estimate HMM parameters
        # Args:
        #     max_iter (int): maximum number of iterations to train
        #     train_data (numpy.ndarray): train data
        #     test_data (numpy.ndarray): test data
        #
        # Returns:
        #     info_dict (dict): dictionary containing information to visualize

        info_dict = {}
        # Initial state distribution
        # pi = np.array([0.25,0.25, 0.25, 0.25])
        pi = np.array([0.5, 0.5])
        # start_time = time.time()

        for it in range(max_iter):

            # Implement the Baum-Welch algorithm here
            # compute in log-domain
            transitions_log = log(self.transitions)
            emissions_log = log(self.emissions)
    
            # Forward pass
            alpha = np.array([-np.inf]*len(train_data)*self.num_states).reshape(len(train_data), self.num_states)
            alpha[0] = log(pi) + emissions_log[:,train_data[0]]

            for t in range(1, len(train_data)):
                # (1*k @ k*k) x (1*k)  => 1*k: computed in log-domain
                alpha[t] = add(logsumexp(add(alpha[t-1].reshape(self.num_states,1), transitions_log), axis=0, keepdims=False),
                               emissions_log[:,train_data[t]])
            
            # Backward pass
            beta = np.array([-np.inf]*len(train_data)*self.num_states).reshape(len(train_data), self.num_states)
            # beta(T) = 1
            beta[-1] = np.array([0]*self.num_states)

            for t in range(len(train_data)-2, -1, -1):
                # log(k*1 @ k*k) + log(k*1) => log(k*1)
                beta[t] = logsumexp(add(add(beta[t+1], emissions_log[:,train_data[t+1]]).reshape(1, self.num_states), transitions_log),
                                    axis=1, keepdims=False)


            # Compute gamma and xi in log-domain
            gamma = alpha + beta
            gamma = gamma - logsumexp(gamma, axis = 1, keepdims=True)
            xi = alpha[:-1].reshape(len(train_data)-1, self.num_states, -1) + \
                transitions_log.reshape(-1, self.num_states, self.num_states) + \
                emissions_log[:,train_data[1:]].T.reshape(len(train_data)-1, -1, self.num_states) + \
                beta[1:].reshape(len(train_data)-1, -1, self.num_states)
            
            xi -= logsumexp(xi, axis=(1,2), keepdims=True)
            
            # update transition matrix
            new_transitions_log = logsumexp(xi, axis = 0, keepdims=False) - \
                          logsumexp(gamma[:-1], axis = 0, keepdims=False).reshape(self.num_states,1)
            
            # update emission matrix
            new_emissions = np.zeros((self.num_states, self.num_outputs))
            for j in range(len(train_data)):
                new_emissions[:,train_data[j]] += exp(gamma[j])
            new_emissions_log = log(new_emissions) - logsumexp(gamma, axis = 0, keepdims=False).reshape(self.num_states,1)

            self.transitions = exp(new_transitions_log)
            self.emissions = exp(new_emissions_log)
            
            # Compute log likelihood of train and test data
            train_ll = self.log_likelihood(train_data)
            test_ll = self.log_likelihood(test_data)

            # # print the parameters
            # if (it+1)%20 == 0:
            #     print(f"*****************Iteration {it+1}*****************")
            #     print("--- %s seconds ---" % (time.time() - start_time))
            #     self.printAB()
            #     print(f"Train log likelihood: {train_ll}")
            #     print(f"Test log likelihood: {test_ll}")

            # save the parameters and log likelihoods
            info_dict[it] = {'train_ll': train_ll, 'test_ll': test_ll, 'B_a': self.emissions[:,0], 'B_n': self.emissions[:,13]}

        return info_dict

    def log_likelihood(self, data):
        # Compute the log likelihood of sequence data
        # Args:
        #     data (numpy.ndarray): 
        #
        # Returns:
        #     prob (float): log likelihood of data
        transitions_log = log(self.transitions)
        emissions_log = log(self.emissions)

        # pi = np.array([0.25,0.25, 0.25, 0.25])
        pi = np.array([0.5, 0.5])
        alpha = np.array([-np.inf]*len(data)*self.num_states).reshape(len(data), self.num_states)
        alpha[0] = log(pi) + emissions_log[:,data[0]]
        
        for t in range(1, len(data)):
                # (1*k @ k*k) x (1*k)  => 1*k: computed in log-domain
                alpha[t] = add(logsumexp(add(alpha[t-1].reshape(self.num_states,1), transitions_log), axis=0, keepdims=False),
                               emissions_log[:,data[t]])
            
        prob = logsumexp(alpha[-1], axis = 0, keepdims=False) / len(data)
        
        return prob

    def visualize(self, info_dict):
        # plot the log likelihood of train and test data
        # Args:
        #     info_dict (dict):
        #
        # Returns:
        #     None
        train_ll = [info_dict[i]['train_ll'] for i in info_dict]
        test_ll = [info_dict[i]['test_ll'] for i in info_dict]
        plt.plot(train_ll, label='train')
        plt.plot(test_ll, label='test')
        plt.legend()
        # plt.savefig('log_likelihood.png', dpi=300)

        # plot the emission probability of 'a' and 'n'
        B_a = np.array([info_dict[i]['B_a'].tolist() for i in info_dict])
        B_n = np.array([info_dict[i]['B_n'].tolist() for i in info_dict])
        plt.figure()
        plt.plot(B_a[:,0], label='q(a|1)')
        plt.plot(B_a[:,1], label='q(a|2)')
        # plt.plot(B_a[:,2], label='q(a|3)') # Q2
        # plt.plot(B_a[:,3], label='q(a|4)') # Q2
 
        plt.plot(B_n[:,0], label='q(n|1)')
        plt.plot(B_n[:,1], label='q(n|2)')
        # plt.plot(B_n[:,2], label='q(n|3)') # Q2
        # plt.plot(B_n[:,3], label='q(n|4)') # Q2
        plt.legend()
        # plt.savefig('emission_prob.png', dpi=300)


def main():
    n_states = 2
    n_outputs = 27
    train_file, test_file = "textA.txt", "textB.txt"
    max_iter = 600

    # define initial transition probability and emission probability
    T_prob, E_prob = get_init_prob_2states() 
    
    # initial the HMM class
    H = HMM(num_states=n_states, num_outputs=n_outputs)

    # initialize HMM with the transition probability and emission probability
    H.initialize(T_prob, E_prob)

    # load text file
    train_data, test_data = load_data(train_file), load_data(test_file)

    # train the parameters of HMM
    info_dict = H.Baum_Welch(max_iter, train_data, test_data)

    # visualize
    H.visualize(info_dict)

if __name__ == "__main__":
    main()