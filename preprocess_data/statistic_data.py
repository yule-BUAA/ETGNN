import json
import itertools


def statistic_data(data_path):
    with open(data_path, 'r') as file:
        users_baskets_dict = json.load(file)

    # users_baskets_dict = {'user_1_id': [[time_1, [item_1_id, item_2_id, item_3_id]], [time_2, [item_5_id]], ...],
    #                       'user_2_id': [[time_1, [item_3_id, item_5_id]], [time_2, [item_1_id, item_2_id]], ...],
    #                       ...}

    items_set, set_count, item_count = set(), 0, 0
    for user_id in users_baskets_dict:
        _, baskets = zip(*users_baskets_dict[user_id])
        set_count += len(baskets)
        item_count += len(list(itertools.chain.from_iterable(baskets)))
        items_set = items_set.union(set(itertools.chain.from_iterable(baskets)))

    # statistics of the dataset
    print(f'number of sets: {set_count}')
    print(f'number of users: {len(users_baskets_dict.keys())}')
    print(f'number of items: {len(items_set)}')
    print(f'number of items per set: {item_count / set_count}')
    print(f'number of sets per user: {set_count / len(users_baskets_dict.keys())}')


if __name__ == "__main__":
    for dataset_name in ['DC', 'TaoBao', 'JingDong', 'TMS']:
        print(f'Statistics on {dataset_name}')
        statistic_data(data_path=f'../dataset/{dataset_name}/{dataset_name}.json')
