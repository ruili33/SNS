# SNS
Code for paper ["Similarity-based Neighbor Selection for Graph LLMs"](https://arxiv.org/pdf/2402.03720.pdf)



### Environment Setup
#### Assume your cuda version is 11.8
```
conda create --name SNS python=3.10
conda activate SNS

conda install pytorch==2.0.0 cudatoolkit=11.8 -c pytorch
conda install -c pyg pytorch-sparse
conda install -c pyg pytorch-scatter
conda install -c pyg pytorch-cluster
conda install -c pyg pyg
pip install ogb
conda install -c dglteam/label/cu118 dgl
pip install transformers
pip install --upgrade accelerate
pip install openai
pip install editdistance
pip install python-json-logger
```

### Dataset Preparation
For `cora`, `pubmed`, `ogbn-arxiv` and `ogbn-product`, please following the dataset processing instructions in [LLM-Structured-Data](https://github.com/TRAIS-Lab/LLM-Structured-Data) to download and place the data under /dataset/{datasetname}.

For `citeseer`, please download via this [link](https://drive.google.com/file/d/16RanD_SHiKtdKP_u1G8ilblUhA9Zkn9J/view?usp=share_link) and place it under /dataset/citeseer. This preprocessed data is borrowed from [Graph-LLM](https://github.com/CurryTang/Graph-LLM).

### API Settings
The api-related arguments are specified in `call_api.py`, including the api key. Please set up your OpenAI API in `call_api.py`.

### Usage

```sh
python main.py  [--dataset DATASET] [--mode MODE] [--k K]
```

### Arguments

- `--dataset`: Specifies the name of dataset. Default is "cora". Choices are ["cora", "pubmed", "citeseer" , "arxiv", "product"]. "arxiv" and "product" represent ogbn-arxiv and ogbn-products respectively.
- `--mode`: Specifies the mode of this execution. Default is "tl". Choices are ['t', 'l', 'tl', "none"]. "t" represents text, while "l" represents label. "tl" represents text+label, and "none" represents vanilla zero-shot, where we don't incorporate graph information.
- `--k`: Specifies the number of neighbors being leveraged. Default is -1, which directly use the k specified in our paper.

### Examples

Run SNS for the cora dataset in text+label mode and default k settings:

```sh
python main.py --dataset cora --mode tl --k -1 
```

Run SNS for the ogbn-arxiv dataset in text only mode and k=8 setting:

```sh
python main.py --dataset arxiv --mode t --k 8 
```

### Citation

If you find the above code is helpful for your research, please cite our paper.

```
@article{li2024similarity,
  title={Similarity-based Neighbor Selection for Graph LLMs},
  author={Li, Rui and Li, Jiwei and Han, Jiawei and Wang, Guoyin},
  journal={arXiv preprint arXiv:2402.03720},
  year={2024}
}
```