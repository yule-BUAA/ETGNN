import pandas as pd
from tqdm import tqdm
import random
import os
import itertools

from utils_preprocess_data import get_frequent_items, reindex_items_users, save_as_json, get_json_data, get_temporal_feature, get_mode_data_dict

frequency_rate = 0.8
min_baskets_length = 4
max_baskets_length = 20
max_basket_boundary = 5


def read_file(data_path: str) -> pd.DataFrame:
    """
        Read original csv files from data_path.

    :param
        data_path : the path of the data.

    :return:
        transaction_df: pd.DataFrame ['customer_id', 'product_id', 'subclass', 'behavior', 'date_time']
    """

    transaction_df = pd.read_csv(data_path, header=None)
    transaction_df.columns = ['customer_id', 'product_id', 'subclass', 'behavior', 'date_time']
    transaction_df = transaction_df[['customer_id', 'subclass', 'behavior', 'date_time']]

    # behavior consists of 'pv'(click), 'buy', 'cart' and 'fav'
    # only buy behavior
    transaction_df = transaction_df[transaction_df['behavior'] == "buy"]

    transaction_df['date_time'] = pd.to_datetime(transaction_df['date_time'], unit='s').astype(str)
    # year-month-day
    transaction_df['date_time'] = transaction_df['date_time'].map(lambda x: x.split(' ')[0])
    transaction_df = transaction_df.sort_values(by='date_time')

    return transaction_df


def generate_baskets(transaction_df: pd.DataFrame, out_sequence_path: str, items_map_dict_path: str, users_map_dict_path: str):
    """
    generate baskets for all the users

    :param transaction_df: pd.DataFrame['customer_id', 'product_id', 'type', 'date_time']
    :param out_sequence_path: str, output sequence path
    :param items_map_dict_path: save path of items id mapping dictionary
    :param users_map_dict_path: save path of users id mapping dictionary
    :return:
        users_baskets_dict: {'user_1_id': [[time_1, [item_1_id, item_2_id, item_3_id]], [time_2, [item_5_id]], ...],
                             'user_2_id': [[time_1, [item_3_id, item_5_id]], [time_2, [item_1_id, item_2_id]], ...],
                             ...}
    """

    random.seed(0)

    # users_baskets = [{'user_1_id': [[time_1, [item_1_id, item_2_id, item_3_id]], [time_2, [item_5_id]], ...]},
    #                  {'user_2_id': [[time_1, [item_3_id, item_5_id]], [time_2, [item_1_id, item_2_id]], ...]},
    #                  ...]
    users_baskets = []
    for user_id, user_df in tqdm(transaction_df.groupby(['customer_id'])):
        baskets_dict = {}
        for day, trans in user_df.groupby(['date_time']):  # select by user and day
            product_index_list = list(set(trans['subclass'].tolist()))
            baskets_dict[day] = product_index_list

        baskets_list = sorted(baskets_dict.items(), key=lambda item: item[0], reverse=False)

        # convert tuple to list, enable the subsequent assignment
        baskets_list = [list(basket) for basket in baskets_list]

        if len(baskets_list) < min_baskets_length:
            continue
        if len(baskets_list) > max_baskets_length:
            baskets_list = baskets_list[:random.randint(max_baskets_length - max_basket_boundary, max_baskets_length)]

        users_baskets.append({user_id: baskets_list})

    # reindex item id and user id
    users_baskets = reindex_items_users(users_baskets, items_map_dict_path, users_map_dict_path)

    # print statistics
    items_set, set_count, item_count = set(), 0, 0
    for user_baskets in users_baskets:
        set_count += len(list(user_baskets.values())[0])
        for basket in list(user_baskets.values())[0]:
            item_count += len(basket[1])
            items_set = items_set.union(basket[1])

    # statistics of the dataset
    print(f'statistic: ')
    print(f'number of sets: {set_count}')
    print(f'number of users: {len(users_baskets)}')
    print(f'number of items: {len(items_set)}')
    print(f'number of items per set: {item_count / set_count}')
    print(f'number of sets per user: {set_count / len(users_baskets)}')
    print(f'date start from {transaction_df["date_time"].min()}, end at {transaction_df["date_time"].max()}')

    # users_baskets_dict = {'user_1_id': [[time_1, [item_1_id, item_2_id, item_3_id]], [time_2, [item_5_id]], ...],
    #                       'user_2_id': [[time_1, [item_3_id, item_5_id]], [time_2, [item_1_id, item_2_id]], ...],
    #                       ...}
    users_baskets_dict = get_json_data(users_baskets)

    save_as_json(data=users_baskets_dict, path=out_sequence_path)

    return users_baskets_dict


def generate_data(users_baskets_dict: dict, out_graph_path_dict: dict, dataset_name: str):
    """

    :param users_baskets_dict: dict，{'user_1_id': [[time_1, [item_1_id, item_2_id, item_3_id]], [time_2, [item_5_id]], ...],
                                      'user_2_id': [[time_1, [item_3_id, item_5_id]], [time_2, [item_1_id, item_2_id]], ...],
                                       ...}
    :param out_graph_path_dict: dict, output graph path for train, validate and test
    :param dataset_name: str, dataset name
    :return:
    """

    num_users = len(users_baskets_dict.keys())

    items_set = set()
    for user_id in users_baskets_dict:
        _, baskets = zip(*users_baskets_dict[user_id])
        items_set = items_set.union(set(itertools.chain.from_iterable(baskets)))
    num_items = len(items_set)

    print(f'number of users: {num_users}')
    print(f'number of items: {num_items}')

    users_baskets_dict = get_temporal_feature(users_baskets_dict, dataset_name=dataset_name)

    for mode in ['train', 'validate', 'test']:
        print(f'Generating {mode} graph...')

        # select data in the train / validate / test
        selected_data_dict = {}
        for user_id in users_baskets_dict:
            if mode == 'train':
                selected_data_dict[user_id] = users_baskets_dict[user_id][:-2]
            elif mode == 'validate':
                selected_data_dict[user_id] = users_baskets_dict[user_id][:-1]
            elif mode == 'test':
                selected_data_dict[user_id] = users_baskets_dict[user_id]
            else:
                raise ValueError(f"mode error for {mode}")

        mode_data_dict = get_mode_data_dict(selected_data_dict, num_classes=num_items)

        save_as_json(data=mode_data_dict, path=out_graph_path_dict[mode])


if __name__ == "__main__":
    data_path = "../original_data/TaoBao_Userbehavior/UserBehavior.csv"

    dataset_name = 'TaoBao'

    root_path = f'../dataset/{dataset_name}'
    os.makedirs(root_path, exist_ok=True)

    items_map_dict_path = f'{root_path}/{dataset_name}_items_map_dic.json'
    users_map_dict_path = f'{root_path}/{dataset_name}_users_map_dic.json'

    out_sequence_path = f'{root_path}/{dataset_name}.json'
    # output path for train, validate and test graphs
    out_graph_path_dict = {
        'train': f'{root_path}/{dataset_name}_train_graph.json',
        'validate': f'{root_path}/{dataset_name}_validate_graph.json',
        'test': f'{root_path}/{dataset_name}_test_graph.json',
    }

    print('Reading files ...\n')
    transaction_df = read_file(data_path)

    print('Removing not frequent items ...\n')
    transaction_df = get_frequent_items(transaction_df, frequency_rate=frequency_rate, key='subclass')

    users_baskets_dict = generate_baskets(transaction_df, out_sequence_path, items_map_dict_path, users_map_dict_path)

    print(f'Generating data file for {dataset_name}...')

    generate_data(users_baskets_dict=users_baskets_dict, out_graph_path_dict=out_graph_path_dict, dataset_name=dataset_name)
