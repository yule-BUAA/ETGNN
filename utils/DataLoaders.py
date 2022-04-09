import json
import os
import torch
import random
from torch.utils.data import Dataset, DataLoader


class CustomizedDataset(Dataset):
    def __init__(self, node_indices_list: list):
        """
        Args:
            node_indices_list: list
        """
        super(CustomizedDataset, self).__init__()

        self.node_indices_list = node_indices_list

    def __getitem__(self, index: int):
        """
        :param index:
        :return:  nodes, tensor (num_nodes, )
                  user_data, list, (baskets, items)
        """
        return self.node_indices_list[index]

    def __len__(self):
        return len(self.node_indices_list)


def get_node_idx_data_loader(node_indices_list: list, batch_size: int, shuffle: bool):
    """
    Args:
        node_indices_list: str
        batch_size: int
        shuffle: boolean
    Returns:
        data_loader: DataLoader
    """

    dataset = CustomizedDataset(node_indices_list)

    data_loader = DataLoader(dataset=dataset,
                             batch_size=batch_size,
                             shuffle=shuffle,
                             drop_last=False)
    return data_loader


class CustomizedDataLoader():
    def __init__(self, graph_folder_path: str, dataset_name: str, data_type: str, sample_neighbors_num: int, hop_num: int):
        """
        Args:
            graph_folder_path: str
            dataset_name: str
            data_type: str, train, validate or test
            sample_neighbors_num: number of sampled neighbors for each node, hop 1 uses the full sample
            hop_num: number of hops
        """

        graph_path = os.path.join(graph_folder_path, f"{dataset_name}_{data_type}_graph.json")

        with open(graph_path) as file:
            # {
            # 'user': {user_id: {item_id: [interact_time_feature, ...], ...},
            # 'item': {item_id: {user_id: [interact_time_feature, ...], ...},
            # 'temporal_feature': {user_id: temporal_feature (list), ...},
            # 'label': {user_id: k_hot_label (list), ...}
            # }
            self.data_dict = json.load(file)

        self.temporal_feature_dimension = len(list(self.data_dict['temporal_feature'].values())[0])
        self.sample_neighbors_num = sample_neighbors_num
        self.hop_num = hop_num

    def get_node_data(self, node_idx: str):
        """

        :param node_idx: central node index, str
        :return:
        """

        # hops_information, list, [[[node_idx, ...], [interact_time, ...]], ...]
        # each hop contains [[node_idx, ...], [interact_time, ...]]
        hops_information = []

        central_node_temporal_feature = torch.Tensor([self.data_dict['temporal_feature'][node_idx]])
        central_node_label = torch.Tensor([self.data_dict['label'][node_idx]])

        last_node_indices = []

        # loop for 1 + hop_num times
        for hop in range(self.hop_num + 1):
            # central node with hop =0
            if hop == 0:
                node_indices = [int(node_idx)]
                node_temporal_features = [self.data_dict['temporal_feature'][node_idx]]

                last_node_indices = [node_idx]
            # hop >= 1
            else:
                node_indices = []
                node_temporal_features = []
                tmp_last_node_indices = []

                # get neighboring users for each node in last_node_indices
                if hop % 2 == 0:
                    for item_idx in last_node_indices:
                        # add to tmp_last_node_indices for next-hop neighbors retrieval with sampling or not
                        if 0 < self.sample_neighbors_num < len(self.data_dict['item'][item_idx].keys()):
                            select_user_idx = random.sample(list(self.data_dict['item'][item_idx].keys()), self.sample_neighbors_num)
                        else:
                            select_user_idx = list(self.data_dict['item'][item_idx].keys())
                        tmp_last_node_indices += select_user_idx

                        for user_idx in select_user_idx:
                            temporal_features_list = self.data_dict['item'][item_idx][user_idx]

                            node_indices += [int(user_idx)] * len(temporal_features_list)
                            node_temporal_features += temporal_features_list

                # get neighboring items for each node in last_node_indices
                else:
                    for user_idx in last_node_indices:
                        # add to tmp_last_node_indices for next-hop neighbors retrieval with sampling (except for the first hop, which uses all the neighbors)
                        if hop != 1 and 0 < self.sample_neighbors_num < len(self.data_dict['user'][user_idx].keys()):
                            select_item_idx = random.sample(list(self.data_dict['user'][user_idx].keys()), self.sample_neighbors_num)
                        else:
                            select_item_idx = list(self.data_dict['user'][user_idx].keys())
                        tmp_last_node_indices += select_item_idx

                        for item_idx in select_item_idx:
                            temporal_features_list = self.data_dict['user'][user_idx][item_idx]

                            node_indices += [int(item_idx)] * len(temporal_features_list)
                            node_temporal_features += temporal_features_list

                # remove duplicated neighboring nodes
                last_node_indices = list(set(tmp_last_node_indices))

            hops_information.append([node_indices, node_temporal_features])

        return hops_information, central_node_temporal_feature, central_node_label

    def get_batch_nodes_data(self, node_indices_list: list):
        """

        :param node_indices_list: central nodes indices, list length -> batch size, each element is a node index (str)
        :return:
        """
        nodes_hops_information, central_nodes_temporal_feature, central_nodes_label = [], [], []

        for node_idx in node_indices_list:
            hops_information, central_node_temporal_feature, central_node_label = self.get_node_data(node_idx=node_idx)
            nodes_hops_information.append(hops_information)
            central_nodes_temporal_feature.append(central_node_temporal_feature)
            central_nodes_label.append(central_node_label)

        # shape -> (1 + hop_num, batch_size)
        hops_nodes_length = []

        # shape, (1 + hop_num, batch_size, max_neighbors_num), each element is a Tensor, with (batch_size, max_neighbors_num)
        hops_nodes_indices = []
        # shape, (1 + hop_num, batch_size, max_neighbors_num, temporal_feature_dimension), each element is a Tensor, with (batch_size, max_neighbors_num, temporal_feature_dimension)
        hops_nodes_temporal_features = []

        # nodes_hops_information, list, shape -> (batch_size, 1 + hop_num, 2, neighbors_num or neighbors_num, temporal_feature_dimension)
        # loop for 1 + hop_num times, use zip* to swap the axis = 0 and axis = 1
        # nodes_hop_information, list, shape -> (batch_size, 2, neighbors_num or neighbors_num, temporal_feature_dimension)
        for nodes_hop_information in zip(*nodes_hops_information):
            # hop_nodes_length, shape -> (batch_size)
            hop_nodes_length = [len(node_hop_information[0]) for node_hop_information in nodes_hop_information]
            max_hop_nodes_length = max(hop_nodes_length)
            # padding for hops information, pad zero for node indices and pad zero-list for time features
            for node_hop_information in nodes_hop_information:
                if len(node_hop_information[0]) < max_hop_nodes_length:
                    pad_length = max_hop_nodes_length - len(node_hop_information[0])
                    node_hop_information[0] += [0] * pad_length
                    node_hop_information[1] += [[0] * self.temporal_feature_dimension] * pad_length
            hops_nodes_length.append(hop_nodes_length)
            # nodes_hop_information, list, shape -> (batch_size, 2, max_neighbors_num or max_neighbors_num, temporal_feature_dimension)
            hop_nodes_indices, hop_nodes_temporal_features = zip(*nodes_hop_information)
            # shape -> (batch_size, max_neighbors_num)
            hops_nodes_indices.append(torch.tensor(hop_nodes_indices))
            # shape -> (batch_size, max_neighbors_num, temporal_feature_dimension)
            hops_nodes_temporal_features.append(torch.Tensor(hop_nodes_temporal_features))

        # Tensor, shape (batch_size, temporal_feature_dimension)
        central_nodes_temporal_feature = torch.cat(central_nodes_temporal_feature, dim=0)
        # Tensor, shape (batch_size, num_items)
        central_nodes_label = torch.cat(central_nodes_label, dim=0)

        return hops_nodes_indices, hops_nodes_temporal_features, hops_nodes_length, central_nodes_temporal_feature, central_nodes_label
