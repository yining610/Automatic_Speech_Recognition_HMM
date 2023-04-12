# Automatic-Speech-Recognition

Directory Structure

```
./project2
├── HMM.py
├── IE_Project2_Report.pdf
├── README.md
├── data
│   ├── clsp.devlbls
│   ├── clsp.devwav
│   ├── clsp.endpts
│   ├── clsp.lblnames
│   ├── clsp.trnlbls
│   ├── clsp.trnscr
│   └── clsp.trnwav
├── data_phonemes
│   └── cmudict-0.7b.txt
├── project2.py
├── project2_accuracy.py
└── project2_pronunciation.py

2 directories, 14 files
```

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

Two new functions in ```project2_accuracy.py``` compared to the file ```project2.py```:
1. The function ```data_split(self, train_ratio=0.8)``` splits the training data to kept and held-out data with ratio $0.8$.
2. The function ```test_val_acc(self, valscr, vallbls)``` evalute the accurcy on the held-out data. 

### Phonemic Baseforms
Command line for reproducing the work in ```IE_Project2_Report.pdf```: ```python project2_pronunciation.py```

Some new functions and variables compared to ```project2.py```
1. The function  ```read_cmu_dict(file_path)``` reads ```cmudict-0.7b.txt``` and creates a dictionary ```self.word2phoneme``` to convert word to its phonemes
2. Replace ```self.letters, self.letter2id, self.id2letter``` by ```self.phonemes, self.phoneme2id, self.id2phoneme``` 
3. ```self.trnscr``` contains phoneme id sequences of the training words.
4. ```self.phoneme_id2hmm``` is constructed in the function ```init_phoneme_hmm(self, lbl_freq, lbl_freq_noise, id2phoneme)```

Still, the phoneme-based HMM was trained 10 epochs and the final log-likelihood is $-388883.0132553676$ which is slightly greater than the result from the grapheme-based HMM.

