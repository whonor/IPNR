# IPNR
This repository contains the code for "Intention-aware User Modeling for Personalized News Recommendation".

## Requirement

- python~=3.8

- torch==1.9.0

- torchtext==0.10.0

- torch-scatter==2.0.9

- torch-geometric==2.0.4

- torch-sparse=0.6.12

- nltk==3.7

- scikit-learn==1.0.2

- pandas==1.4.3

- numpy==1.23.0



## Dataset Preparation
The experiments are conducted on the MIND dataset. Our code will try to download and sample the MIND dataset to the directory `../MIND`.

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

<br/><br/>


## Run

<pre><code>python main.py --news_encoder=IPNR --user_encoder=IPNR</code></pre>


### Credits

- Dataset by **MI**crosoft **N**ews **D**ataset (MIND), see <https://msnews.github.io/>.
- Reference https://github.com/Veason-silverbullet/NNR
