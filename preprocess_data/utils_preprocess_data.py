import pandas as pd
import json
from tqdm import tqdm
import datetime
import torch
from datetime import datetime
from collections import defaultdict


def get_frequent_items(transaction_df: pd.DataFrame, frequency_rate: float, key: str):
    """
    get frequent items based on frequency_rate

    :param transaction_df: pd.DataFrame
    :param frequency_rate
    :param key
    :return:
        new_df: pd.DataFrame
    """
    value_counts = transaction_df[key].value_counts()
    total_number = len(transaction_df)
    sum_number = 0
    item_list = []
    for index in tqdm(value_counts.index):
        if sum_number / total_number >= frequency_rate:
            break
        sum_number += value_counts[index]
        item_list.append(index)

    new_df = transaction_df[transaction_df[key].isin(item_list)]
    return new_df


def save_as_json(data: dict or list, path: str):
    """
    save data as json file with path

    :param data:
    :param path:
    :return:
    """
    with open(path, "w") as file:
        file.write(json.dumps(data))
        file.close()
        print(f'{path} writes successfully.')


def reindex_items_users(users_baskets: list, items_map_dict_path: str, users_map_dict_path: str):
    """
    reindex item id and user id in baskets

    :param users_baskets: list, [{'user_1_id': [[time_1, [item_1_id, item_2_id, item_3_id]], [time_2, [item_5_id]], ...]},
                                 {'user_2_id': [[time_1, [item_3_id, item_5_id]], [time_2, [item_1_id, item_2_id]], ...]},
                                 ...]

    :param items_map_dict_path: save path of items id mapping dictionary
    :param users_map_dict_path: save path of users id mapping dictionary
    :return: users_baskets, list, which completes the mapping of items id and users id
    """

    # reindex item id
    items_list = []
    for user_baskets in users_baskets:
        # basket is a list [time, [item_1_id, item_2_id, ...]]
        for basket in list(user_baskets.values())[0]:
            items_list.extend(basket[1])

    unique_item_id_list = list(set(items_list))
    unique_item_id_list.sort()

    # generate item reindex mapping
    item_id_map_dict = {}
    for index, value in enumerate(unique_item_id_list):
        item_id_map_dict[value] = index
    save_as_json(item_id_map_dict, items_map_dict_path)

    # reindex item
    for user_baskets in users_baskets:
        for basket in list(user_baskets.values())[0]:
            for index, item_id in enumerate(basket[1]):
                basket[1][index] = item_id_map_dict[item_id]

    # reindex user id
    user_id_map_dict = {}
    users_baskets_copy = []
    for index, user_baskets in enumerate(users_baskets):
        user_id_map_dict[list(user_baskets.keys())[0]] = index
        users_baskets_copy.append({index: list(user_baskets.values())[0]})

    save_as_json(user_id_map_dict, users_map_dict_path)

    return users_baskets_copy


def get_json_data(data_list: list):
    """
    convert data_list into json format

    :param data_list:
    :return:
    """
    users_baskets_dict = {}
    for user_baskets_dict in data_list:
        users_baskets_dict.update(user_baskets_dict)
    return users_baskets_dict


def get_temporal_feature(users_baskets_dict: dict, dataset_name: str):
    """

    :param users_baskets_dict: dict，{'user_1_id': [[time_1, [item_1_id, item_2_id, item_3_id]], [time_2, [item_5_id]], ...],
                                      'user_2_id': [[time_1, [item_3_id, item_5_id]], [time_2, [item_1_id, item_2_id]], ...],
                                       ...}
    :param dataset_name: str, dataset name

    :return: users_baskets_dict, users baskets dictionary with temporal feature
    """
    for user_id in users_baskets_dict:
        # list of user baskets
        user_baskets = users_baskets_dict[user_id]
        # user end time, could be list or datetime
        if dataset_name == 'DC':
            user_end_time = user_baskets[-1][0][1]
        elif dataset_name == 'TaoBao' or dataset_name == 'JingDong':
            user_end_time = datetime.strptime(user_baskets[-1][0], '%Y-%m-%d')
        else:
            raise ValueError(f'wrong dataset name {dataset_name}!')

        for basket in user_baskets:
            set_time = basket[0]
            # time_feature is a list
            if dataset_name == 'DC':
                week, day = set_time
                delta_t = user_end_time - day
                time_feature = [week, day, delta_t]
            elif dataset_name == 'TaoBao' or dataset_name == 'JingDong':
                date = datetime.strptime(set_time, '%Y-%m-%d')
                # Monday is 0 and Sunday is 6
                weekday = date.weekday()
                is_holiday = 0 if 0 <= weekday <= 4 else 1
                delta_t = (user_end_time - date).days
                time_feature = [date.year, date.month, date.day, weekday, is_holiday, delta_t]
            else:
                raise ValueError(f'wrong dataset name {dataset_name}!')

            basket[0] = time_feature

    return users_baskets_dict


def get_k_hot_encoding(id_list: list, num_classes: int):
    """
    get k-hot encoding based on the input ids
    :param id_list: list, list of ids, shape (input_items_num, )
    :param num_classes:
    :return:
        k_hot_encoding[i] = 1 if i in id_list, else 0
    """
    k_hot_encoding = torch.zeros(num_classes)
    if len(id_list) > 0:
        k_hot_encoding[id_list] = 1
    return k_hot_encoding.tolist()


def get_mode_data_dict(selected_data_dict: dict, num_classes: int):
    """
    get mode data dictionary
    :param selected_data_dict: dict, selected data dictionary
    :param num_classes:
    :return:
        mode_data_dict: dict
    """
    mode_data_dict = defaultdict(dict)

    for user_id in tqdm(selected_data_dict):
        user_baskets = selected_data_dict[user_id]
        for basket in user_baskets[:-1]:
            basket_temporal_feature = basket[0]
            for item_id in basket[1]:
                if mode_data_dict['user'].get(user_id) is None:
                    mode_data_dict['user'][user_id] = defaultdict(dict)
                if mode_data_dict['item'].get(item_id) is None:
                    mode_data_dict['item'][item_id] = defaultdict(dict)

                if mode_data_dict['user'][user_id].get(item_id) is None:
                    mode_data_dict['user'][user_id][item_id] = []
                if mode_data_dict['item'][item_id].get(user_id) is None:
                    mode_data_dict['item'][item_id][user_id] = []

                mode_data_dict['user'][user_id][item_id].append(basket_temporal_feature)
                mode_data_dict['item'][item_id][user_id].append(basket_temporal_feature)

        mode_data_dict['temporal_feature'][user_id] = user_baskets[-1][0]
        mode_data_dict['label'][user_id] = get_k_hot_encoding(user_baskets[-1][1], num_classes=num_classes)

    return mode_data_dict
