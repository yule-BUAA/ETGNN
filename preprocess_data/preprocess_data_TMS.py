from tqdm import tqdm
import random
import json
import os
import itertools

from utils_preprocess_data import reindex_items_users, save_as_json, get_json_data, get_mode_data_dict

min_baskets_length = 4
max_baskets_length = 20
max_basket_boundary = 5


def generate_baskets(data_dict: dict, out_sequence_path: str, items_map_dict_path: str, users_map_dict_path: str):
    """
    generate baskets for all the users

    :param data_dict: data dictionary {
                                       'train' : {user_1_id: [[item_1_id, item_2_id], [item_3_id, item_5_id], ...], ...},
                                       'validate' : {user_1_id: [[item_3_id, item_5_id], [item_1_id, item_4_id], ...], ...},
                                       'test' : {user_1_id: [[item_1_id], [item_2_id, item_3_id, item_7_id], ...], ...}
                                      }
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
    
    for mode in ['train', 'validate', 'test']:
        for user_id, user_baskets in tqdm(data_dict[mode].items()):
            user_id = f'{mode}_{user_id}'
            sequence_length = len(user_baskets)
            for index, basket in enumerate(user_baskets):
                time_feature = [sequence_length - 1 - index]
                user_baskets[index] = [time_feature, basket]

            baskets_list = [basket for basket in user_baskets if len(basket[1]) > 0]

            if len(baskets_list) < min_baskets_length:       # remove too short baskets
                continue
            elif len(baskets_list) > max_baskets_length:     # cut too long baskets
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

    # users_baskets_dict = {'user_1_id': [[time_1, [item_1_id, item_2_id, item_3_id]], [time_2, [item_5_id]], ...],
    #                       'user_2_id': [[time_1, [item_3_id, item_5_id]], [time_2, [item_1_id, item_2_id]], ...],
    #                       ...}
    users_baskets_dict = get_json_data(users_baskets)

    save_as_json(data=users_baskets_dict, path=out_sequence_path)

    return users_baskets_dict


def generate_data(users_baskets_dict: dict, out_graph_path_dict: dict):
    """

    :param users_baskets_dict: dict，{'user_1_id': [[time_1, [item_1_id, item_2_id, item_3_id]], [time_2, [item_5_id]], ...],
                                      'user_2_id': [[time_1, [item_3_id, item_5_id]], [time_2, [item_1_id, item_2_id]], ...],
                                       ...}
    :param out_graph_path_dict: dict, output graph path for train, validate and test
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

    # since TMS only has periodic temporal feature, there is not need to call get_temporal_feature function
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
    data_path = "../original_data/SOS_Data/tags-math-sx-seqs.json"

    dataset_name = 'TMS'

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
    with open(data_path, 'r') as file:
        data_dict = json.load(file)

    users_baskets_dict = generate_baskets(data_dict, out_sequence_path, items_map_dict_path, users_map_dict_path)

    print(f'Generating data file for {dataset_name}...')

    generate_data(users_baskets_dict=users_baskets_dict, out_graph_path_dict=out_graph_path_dict)
