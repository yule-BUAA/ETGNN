import torch
import os
import warnings

from model.ETGNN import ETGNN
from utils.metric import evaluate_model
from utils.load_config import get_users_items_num
from utils.utils import convert_to_gpu, get_n_params, set_random_seed
from utils.DataLoaders import CustomizedDataLoader, get_node_idx_data_loader

config = {
        "seed": 0,
        "model_name": "ETGNN",
        "dataset_name": "DC",
        "sample_neighbors_num": 10,
        "cuda": 0,
        "DC": {
            "learning_rate": 0.001,
            "embedding_dimension": 64,
            "embedding_dropout": 0.2,
            "temporal_attention_dropout": 0.5,
            "hop_num": 3,
            "temporal_feature_importance": 0.3,
            "batch_size": 8
        },
        "TaoBao": {
            "learning_rate": 0.001,
            "embedding_dimension": 32,
            "embedding_dropout": 0.0,
            "temporal_attention_dropout": 0.5,
            "hop_num": 3,
            "temporal_feature_importance": 0.05,
            "batch_size": 32
        },
        "JingDong": {
            "learning_rate": 0.001,
            "embedding_dimension": 64,
            "embedding_dropout": 0.2,
            "temporal_attention_dropout": 0.5,
            "hop_num": 3,
            "temporal_feature_importance": 0.01,
            "batch_size": 8
        },
        "TMS": {
            "learning_rate": 0.001,
            "embedding_dimension": 64,
            "embedding_dropout": 0.3,
            "temporal_attention_dropout": 0.5,
            "hop_num": 2,
            "temporal_feature_importance": 1.0,
            "batch_size": 32
        }
}


def get_attribute(attribute_name: str, default_value=None):
    """
    get attribute in config
    :param attribute_name:
    :param default_value:
    :return:
    """
    try:
        return config[attribute_name]
    except KeyError:
        return default_value

# dataset specified settings
config.update(config[f"{get_attribute('dataset_name')}"])
config.pop('DC')
config.pop('TaoBao')
config.pop('JingDong')
config.pop('TMS')

config['graph_folder_path'] = f"{os.path.dirname(os.path.dirname(__file__))}/dataset/{get_attribute('dataset_name')}"
config['sequence_data_path'] = f"{os.path.dirname(os.path.dirname(__file__))}/dataset/{get_attribute('dataset_name')}/{get_attribute('dataset_name')}.json"
config['num_users'], config['num_items'] = get_users_items_num(get_attribute('sequence_data_path'))
config['device'] = f'cuda:{get_attribute("cuda")}' if torch.cuda.is_available() and get_attribute("cuda") >= 0 else 'cpu'


if __name__ == '__main__':
    warnings.filterwarnings('ignore')

    set_random_seed(get_attribute('seed'))

    print('getting data loaders...')
    node_idx_data_loader = get_node_idx_data_loader([str(i) for i in range(get_attribute('num_users'))],
                                                    batch_size=get_attribute('batch_size'), shuffle=True)

    train_data_loader = CustomizedDataLoader(graph_folder_path=get_attribute('graph_folder_path'),
                                             dataset_name=get_attribute('dataset_name'), data_type='train',
                                             sample_neighbors_num=get_attribute('sample_neighbors_num'),
                                             hop_num=get_attribute('hop_num'))
    valid_data_loader = CustomizedDataLoader(graph_folder_path=get_attribute('graph_folder_path'),
                                             dataset_name=get_attribute('dataset_name'), data_type='validate',
                                             sample_neighbors_num=get_attribute('sample_neighbors_num'),
                                             hop_num=get_attribute('hop_num'))
    test_data_loader = CustomizedDataLoader(graph_folder_path=get_attribute('graph_folder_path'),
                                            dataset_name=get_attribute('dataset_name'), data_type='test',
                                            sample_neighbors_num=get_attribute('sample_neighbors_num'),
                                            hop_num=get_attribute('hop_num'))

    print('creating model...')
    model = ETGNN(num_items=get_attribute('num_items'), num_users=get_attribute('num_users'),
                  hop_num=get_attribute('hop_num'), embedding_dimension=get_attribute('embedding_dimension'),
                  temporal_feature_dimension=train_data_loader.temporal_feature_dimension,
                  embedding_dropout=get_attribute('embedding_dropout'),
                  temporal_attention_dropout=get_attribute('temporal_attention_dropout'),
                  temporal_information_importance=get_attribute('temporal_feature_importance'))

    save_model_path = f"../save_model_folder/{get_attribute('dataset_name')}/{get_attribute('model_name')}/{get_attribute('model_name')}.pkl"

    model.load_state_dict(torch.load(save_model_path, map_location='cpu'), strict=True)

    model = convert_to_gpu(model, device=get_attribute('device'))

    print(model)

    print(f'Model #Params: {get_n_params(model)}.')

    data_loader_dic = {"train": train_data_loader, "validate": valid_data_loader, "test": test_data_loader}

    print(f'getting final performance on dataset {get_attribute("dataset_name")}...')

    train_metrics = evaluate_model(model, node_idx_data_loader, data_loader_dic['train'], device=get_attribute('device'))
    valid_metrics = evaluate_model(model, node_idx_data_loader, data_loader_dic['validate'], device=get_attribute('device'))
    test_metrics = evaluate_model(model, node_idx_data_loader, data_loader_dic['test'], device=get_attribute('device'))

    print(f'train metrics -> {train_metrics}')
    print(f'validate metrics -> {valid_metrics}')
    print(f'test metrics -> {test_metrics}')
