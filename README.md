# IPNR
This repository is code for "Intention-aware User Modeling for Personalized News Recommendation".

## Requirement

python~=3.8

torc==1.9.0

torchtext==0.10.0

torch-scatter==2.0.9

nltk==3.7

scikit-learn==1.0.2


## Dataset

```bash
# Download GloVe pre-trained word embedding
https://nlp.stanford.edu/data/glove.840B.300d.zip

# Download MIND dataset
https://msnews.github.io/.
```

## Run

```bash
# Train the model, meanwhile save checkpoints
python3 src/train1.py
# Load latest checkpoint and evaluate on the test set
python3 src/evaluate.py
```

### Credits

- Dataset by **MI**crosoft **N**ews **D**ataset (MIND), see <https://msnews.github.io/>.
- Reference https://github.com/Veason-silverbullet/NNR
