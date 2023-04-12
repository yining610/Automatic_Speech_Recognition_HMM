# Automatic-Speech-Recognition

## Primary System
Train and test the model: ```python project2.py```

Hyparameters:

1. Number of epochs: ```num_epochs=10```
2. Initial forward probability $\alpha_0$: ```init_prob = np.asarray([1] + [0] * (word_hmm.num_states - 1), dtype=np.float64)```
3. Fixed null arc transition probabilities during training: $0.2$ for letter HMM ```letter_id2hmm[letter_id].exiting_prob = 0.2```, $0.25$ for SIL HMM ```letter_id2hmm[noise_id].exiting_prob = 0.25```

## Contrastive System
### Held-out Test
Find the $N^{\ast}$: ```python project2_accuracy.py --find_N --num_epochs 10```

Train and test the model with $N^\ast = 3$: ```python project2_accuracy.py --num_epochs 3```

The best held-out accuracy is $0.7215189873417721$ after 3 epochs.

Two new functions in ```project2_accuracy.py``` compare to the file ```project2.py```:
1. The function ```data_split(self, train_ratio=0.8)``` splits the training data to kept and held-out data with ratio $0.8$.
2. The function ```test_val_acc(self, valscr, vallbls)``` evalute the accurcy on the held-out data. 

