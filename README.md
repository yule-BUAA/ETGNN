# Element-guided Temporal Graph Representation Learning for Temporal Sets Prediction

The description of "Element-guided Temporal Graph Representation Learning for Temporal Sets Prediction" 
is [available here](https://dl.acm.org/doi/10.1145/3485447.3512064). 

### Original data:
The original data could be downloaded from [here](https://drive.google.com/file/d/1f2Eexc9vwRYYrrvLzuL4zBnWwWs6EHhI/view?usp=sharing). 
You can download the data and then put the data files in the ```./original_data``` folder.


### To run the code:
  
  1. run ```./preprocess_data/preprocess_data_{dataset_name}.py``` to preprocess the original data, 
     where ```dataset_name``` could be DC, TaoBao, JingDong and TMS. 
     We also provide the preprocessed datasets at [here](https://drive.google.com/file/d/1Maal10-7LCLO-1kDl7per82f7nRD_Pmi/view?usp=sharing), 
     which should be put in the ```./dataset``` folder.
     
  2. run ```./train/train_ETGNN.py``` to train the model on different datasets using the configuration in ```./utils/config.json```.

  3. run ```./evaluate/evaluate_ETGNN.py``` to evaluate the model. 
     Please make sure the ```config``` in ```evaluate_ETGNN.py``` keeps identical to that in the model training process.


## Environments:
- [PyTorch 1.8.1](https://pytorch.org/)
- [tqdm](https://github.com/tqdm/tqdm)
- [numpy](https://numpy.org/)
- [pandas](https://pandas.pydata.org/)


## Hyperparameter settings:
Hyperparameters can be found in ```./utils/config.json``` file, and you can adjust them when training the model on different datasets.

| Hyperparameters  | DC  | TaoBao  | JingDong  | TMS |
| -------    | ------- | -------  | -------  | -------  |
| learning rate  | 0.001  | 0.001  | 0.001  |  0.001   |
| embedding dimension  | 64  | 32  | 64  |  64   |
| embedding dropout  | 0.2  | 0.0  | 0.2  |  0.3   |
| temporal attention dropout  | 0.5  | 0.5  | 0.5  |  0.5   |
| number of hops  | 3  | 3  | 3 |  2  |
| temporal information importance  | 0.3  | 0.05  | 0.01  |  1.0   |


## Citation (To be Updated)
Please consider citing our paper when using the codes or datasets.

```
@inproceedings{DBLP:conf/www/xxxx,
  author    = {Le Yu and
               Guanghui Wu and
               Leilei Sun and
               Bowen Du and
               Weifeng Lv},
  title     = {Element-guided Temporal Graph Representation Learning for Temporal Sets Prediction},
  booktitle = {{WWW} '22: The ACM Web Conference 2022, Virtual Event, Lyon, France, April 25--29, 2022},
  pages     = {xxx--xxx},
  publisher = {{ACM}},
  year      = {2022}
}
```
