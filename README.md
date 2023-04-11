# Automatic-Speech-Recognition

## Primary System
Train and test the model: ```python project2.py```

Hyparameters:

1. Number of epochs: ```num_epochs=10```
2. Initial forward probability $\alpha_0$: ```init_prob = np.asarray([1] + [0] * (word_hmm.num_states - 1), dtype=np.float64)```
3. Fixed null arc transition probabilities during training: $0.2$ for letter HMM ```letter_id2hmm[letter_id].exiting_prob = 0.2```, $0.25$ for SIL HMM ```letter_id2hmm[noise_id].exiting_prob = 0.25```
