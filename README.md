# IPNR
This repository contains the code for "Intention-aware User Modeling for Personalized News Recommendation".

## Requirement

python~=3.8

torch==1.9.0

torchtext==0.10.0

torch-scatter==2.0.9

torch-geometric==1.6.1

nltk==3.7

scikit-learn==1.0.2



## Dataset Preparation
The experiments are conducted on the 200k-MIND dataset. Our code will try to download and sample the 200k-MIND dataset to the directory `../MIND-200k` (see Line 140 of `config.py` and `prepare_MIND_dataset.py`).

Since the MIND dataset is quite large, if our code cannot download it successfully due to unstable network connection, please execute the shell file `download_extract_MIND.sh` instead. If the automatic download still fails, we recommend to download the MIND dataset and knowledge graph manually according to the links in `download_extract_MIND.sh`.

Assume that now the pwd is `./IPNR`, the downloaded and extracted MIND dataset should be organized as

    (terminal) $ bash download_extract_MIND.sh # Assume this command is executed successfully
    (terminal) $ cd ../MIND
    (terminal) $ tree -L 2
    (terminal) $ .
                 ├── dev
                 │   ├── behaviors.tsv
                 │   ├── entity_embedding.vec
                 │   ├── news.tsv
                 │   ├── __placeholder__
                 │   └── relation_embedding.vec
                 ├── dev.zip
                 ├── train
                 │   ├── behaviors.tsv
                 │   ├── entity_embedding.vec
                 │   ├── news.tsv
                 │   ├── __placeholder__
                 │   └── relation_embedding.vec
                 ├── train.zip
    
<br/>

Then run prepare_MIND-dataset.py to preprocess the data.
## Environment Requirements
    (terminal) $ pip install -r requirements.txt


<br/><br/>


## Experiment Running
<hr>Our Model
<pre><code>python main.py --news_encoder=IPNR --user_encoder=IPNR</code></pre>

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
