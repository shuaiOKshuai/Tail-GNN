
# Tail-GNN: Tail-Node Graph Neural Networks 
We provide the implementaion for our paper "Tail-GNN: Tail-Node Graph Neural Networks", which is published in KDD-2021.


## 1. Desription
The repository is organised as follows:

* dataset/: contains our benchmark datasets: email, squirrel, actor, cs-citation, amazon. All datasets will be processed on the fly. 
  * For email, we use data split as meta-tail2vec. 
  * For large dataset as cs-citation, it may take 5+ mins to process before training will start in first run. 
  * For amazon, please download dataset from [this link](https://github.com/pyyush/GraphML) and put in this folder before running. It should be run on GPU with 16GB memory.

* models/: contains our model.
  * tailgnn.py: implementation of our tail_gnn model.
  * tailgnn_sp.py: our version for large dataset.

* layers/: contains component layers for our model.  
* utils/: contains tools for preprocessing data, metrics for evaluation, etc.
* link_prediction/: sub-directory to run the link prediction task.
  

## 2. Requirements
To install required packages
- pip3 install -r requirements.txt

## 3. Running experiments

### Tail node classification:
- python3 main.py --dataset=squirrel --eta=0.1 --mu=0.001 --k=5
  
For larger datasets such as cs-citation, please use the sparse version:
- python3 main_sp.py --dataset=cs-citation


### Link prediction:
- cd link_prediction/
- python3 main.py --dataset=squirrel 

For larger datasets:
- python3 main_sp.py --dataset=cs-citation


### Note:
- We utilize utils/data_process.py to prepare different datasets into the input format. To run model on your datasets, please refer to utils/data_process.py to process data to input format.
- For Email dataset, we utilize the same data split as reported in meta-tail2vec.


## Cite
