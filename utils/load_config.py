import os
import json
import itertools
import torch

abs_path = os.path.join(os.path.dirname(__file__), "config.json")
with open(abs_path) as file:
    config = json.load(file)


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


def get_users_items_num(data_path: str):
    """
    get number of users and items
    :param data_path:
    :return:
    """
    with open(data_path, 'r') as file:
        users_baskets_dict = json.load(file)

    # users_baskets_dict = {'user_1_id': [[time_1, [item_1_id, item_2_id, item_3_id]], [time_2, [item_5_id]], ...],
    #                       'user_2_id': [[time_1, [item_3_id, item_5_id]], [time_2, [item_1_id, item_2_id]], ...],
    #                       ...}

    num_users = len(users_baskets_dict.keys())
    items_set = set()
    for user_id in users_baskets_dict:
        _, baskets = zip(*users_baskets_dict[user_id])
        items_set = items_set.union(set(itertools.chain.from_iterable(baskets)))
    num_items = len(items_set)

    return num_users, num_items

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
