
# Tail-GNN: Tail-Node Graph Neural Networks 
We provide the code and datasets for our paper "Tail-GNN: Tail-Node Graph Neural Networks" (Tail-GNN for short), which is published in KDD-2021.


## 1. Desription
The repository is organised as follows:

* dataset/: contains the benchmark datasets: email, squirrel, actor, cs-citation and amazon. All datasets will be processed on the fly. 
  * For dataset Email, we use the same data split in paper [meta-tail2vec](https://github.com/shuaiOKshuai/meta-tail2vec). 
  * For large dataset cs-citation, when running the model it may take around 5+ mins to process the dataset before training starts. 
  * For dataset Amazon, due to its large size, we don't include it in this folder. Please download it from [this link](https://github.com/pyyush/GraphML) and put it into this folder before running. It should be run on GPU with 16GB memory. Note that we utilize a ~1M size graph in the experiments (a connected subgraph), not the original ~2M one.

* models/: contains our model Tail-GNN.
  * tailgnn.py: implementation of Tail-GNN model.
  * tailgnn_sp.py: sparse version of Tail-GNN for large dataset.

* layers/: contains the model layers of Tail-GNN.  
* utils/: contains tool functions for preprocessing data, and metrics for evaluation, etc.
* link_prediction/: sub-directory for codes of link prediction task.
  

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
- We utilize utils/data_process.py to prepare different datasets into the input format. To run the code on your datasets, please refer to utils/data_process.py to process the corresponding datasets into the input format.


## 4. Cite

	@inproceedings{liu2021tailgnn,
		title={Tail-GNN: Tail-Node Graph Neural Networks},
		author={Liu, Zemin and Nguyen, Trung-Kien and Fang, Yuan},
		booktitle={Proceedings of the 27th ACM SIGKDD International Conference on Knowledge Discovery & Data Mining},
		year={2021}
	}
