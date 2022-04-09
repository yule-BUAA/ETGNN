import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class PeriodicTimeEncoder(nn.Module):
    def __init__(self, embedding_dimension: int):
        super(PeriodicTimeEncoder, self).__init__()

        self.embedding_dimension = embedding_dimension
        self.scale_factor = (1 / (embedding_dimension // 2)) ** 0.5

        self.w = nn.Parameter(torch.randn(1, embedding_dimension // 2))
        self.b = nn.Parameter(torch.randn(1, embedding_dimension // 2))

    def forward(self, input_relative_time: torch.Tensor):
        """

        :param input_relative_time: shape (batch_size, temporal_feature_dimension) or (batch_size, max_neighbors_num, temporal_feature_dimension)
               input_time_dim = 1 since the feature denotes relative time (scalar)
        :return:
            time_encoding, shape (batch_size, embedding_dimension) or (batch_size, max_neighbors_num, embedding_dimension)
        """

        # cos_encoding, shape (batch_size, embedding_dimension // 2) or (batch_size, max_neighbors_num, embedding_dimension // 2)
        cos_encoding = torch.cos(torch.matmul(input_relative_time, self.w) + self.b)
        # sin_encoding, shape (batch_size, embedding_dimension // 2) or (batch_size, max_neighbors_num, embedding_dimension // 2)
        sin_encoding = torch.sin(torch.matmul(input_relative_time, self.w) + self.b)

        # time_encoding, shape (batch_size, embedding_dimension) or (batch_size, max_neighbors_num, embedding_dimension)
        time_encoding = self.scale_factor * torch.cat([cos_encoding, sin_encoding], dim=-1)

        return time_encoding


class ETGNN(nn.Module):

    def __init__(self, num_items: int, num_users: int, hop_num: int, embedding_dimension: int, temporal_feature_dimension: int,
                 embedding_dropout: float, temporal_attention_dropout: float, temporal_information_importance: float):
        """

        :param num_items: int, number of items
        :param num_users: int, number of users
        :param hop_num: int, , number of hops
        :param embedding_dimension: int, dimension of embedding
        :param temporal_feature_dimension: int, the input dimension of temporal feature
        :param embedding_dropout: float, embedding dropout rate
        :param temporal_attention_dropout: float, temporal attention dropout rate
        :param temporal_information_importance: float, importance of temporal information
        """
        super(ETGNN, self).__init__()

        self.num_items = num_items
        self.num_users = num_users
        self.hop_num = hop_num
        self.embedding_dimension = embedding_dimension
        self.temporal_feature_dimension = temporal_feature_dimension
        self.embedding_dropout = embedding_dropout
        self.temporal_attention_dropout = temporal_attention_dropout
        self.temporal_information_importance = temporal_information_importance

        self.items_embedding = nn.Embedding(num_items, embedding_dimension)
        self.users_embedding = nn.Embedding(num_users, embedding_dimension)

        self.leaky_relu_func = nn.LeakyReLU(negative_slope=0.2)

        self.embedding_dropout = nn.Dropout(embedding_dropout)
        self.temporal_attention_dropout = nn.Dropout(temporal_attention_dropout)

        # the TMS dataset only has periodic temporal information, and thus does not need the semantic temporal information encoder
        if temporal_feature_dimension > 1:
            self.semantic_time_encoder = nn.Linear(temporal_feature_dimension - 1, out_features=embedding_dimension//2)
            self.periodic_time_encoder = PeriodicTimeEncoder(embedding_dimension=embedding_dimension//2)
        else:
            self.periodic_time_encoder = PeriodicTimeEncoder(embedding_dimension=embedding_dimension)

        self.fc_projection = nn.Linear(hop_num * embedding_dimension, embedding_dimension)

    def forward(self, hops_nodes_indices: list, hops_nodes_temporal_features: list, hops_nodes_length: list, central_nodes_temporal_feature: torch.Tensor):
        """

        :param hops_nodes_indices: list, shape (1 + hop_num, batch_size, max_neighbors_num)
        hop: 0 -> self, 1 -> 1 hop, 2 -> 2 hop ..., odd number -> item, even number -> user
        :param hops_nodes_temporal_features: list, shape, (1 + hop_num, batch_size, max_neighbors_num, temporal_feature_dimension)
        :param hops_nodes_length: list, shape (1 + hop_num, batch_size)
        :param central_nodes_temporal_feature: Tensor, shape (batch_size, temporal_feature_dimension)
        :return:
            set_prediction, shape (batch_size, num_items),
        """
        # shape (num_items, embedding_dimension)
        query_embeddings = self.embedding_dropout(self.items_embedding(
            torch.tensor([i for i in range(self.num_items)]).to(central_nodes_temporal_feature.device)))

        # list, shape (1 + hop_num, batch_size, embedding_dimension)
        nodes_hops_embedding = []

        if self.temporal_feature_dimension > 1:
            # shape (batch_size, embedding_dimension // 2)
            central_nodes_semantic_time_feature = self.semantic_time_encoder(central_nodes_temporal_feature[:, :-1])
            # shape (batch_size, embedding_dimension // 2)
            central_nodes_periodic_time_feature = self.periodic_time_encoder(central_nodes_temporal_feature[:, -1].unsqueeze(dim=-1))
            # shape (batch_size, embedding_dimension)
            central_nodes_time_embedding = torch.cat([central_nodes_semantic_time_feature, central_nodes_periodic_time_feature], dim=-1)
        else:
            # shape (batch_size, embedding_dimension)
            central_nodes_time_embedding = self.periodic_time_encoder(central_nodes_temporal_feature)

        for hop_index in range(len(hops_nodes_indices)):
            # hop_nodes_indices -> tensor (batch_size, max_neighbors_num)
            # hop_nodes_temporal_features -> Tensor (batch_size, max_neighbors_num, temporal_feature_dimension)
            hop_nodes_indices, hop_nodes_temporal_features = hops_nodes_indices[hop_index], hops_nodes_temporal_features[hop_index]

            # user
            if hop_index % 2 == 0:
                # skip central node itself feature
                if hop_index == 0:
                    continue
                else:
                    # shape (batch_size, max_neighbors_num, embedding_dimension)
                    hop_nodes_embedding = self.users_embedding(hop_nodes_indices)
            # item
            else:
                # shape (batch_size, max_neighbors_num, embedding_dimension)
                hop_nodes_embedding = self.items_embedding(hop_nodes_indices)

            hop_nodes_embedding = self.embedding_dropout(hop_nodes_embedding)

            # shape (batch_size, num_items, max_neighbors_num),  (num_items, embedding_dimension) einsum (batch_size, max_neighbors_num, embedding_dimension)
            attention = torch.einsum('if,bnf->bin', query_embeddings, hop_nodes_embedding)

            # mask based on hops_nodes_length, shape (batch_size, num_items, max_neighbors_num)
            attention_mask = torch.zeros_like(attention)
            for node_idx, hop_node_length in enumerate(hops_nodes_length[hop_index]):
                attention_mask[node_idx][:, hop_node_length:] = - np.inf

            if self.temporal_feature_dimension > 1:
                # shape (batch_size, max_neighbors_num, embedding_dimension // 2)
                hop_nodes_semantic_time_feature = self.semantic_time_encoder(hop_nodes_temporal_features[:, :, :-1])
                # shape (batch_size, max_neighbors_num, embedding_dimension // 2)
                hop_nodes_periodic_time_feature = self.periodic_time_encoder(hop_nodes_temporal_features[:, :, -1].unsqueeze(dim=-1))
                # shape (batch_size, max_neighbors_num, embedding_dimension)
                hop_nodes_time_embedding = torch.cat([hop_nodes_semantic_time_feature, hop_nodes_periodic_time_feature], dim=-1)
            else:
                # shape (batch_size, max_neighbors_num, embedding_dimension)
                hop_nodes_time_embedding = self.periodic_time_encoder(hop_nodes_temporal_features)

            # shape (batch_size, num_items, max_neighbors_num),  (batch_size, num_items, embedding_dimension) einsum (batch_size, max_neighbors_num, embedding_dimension)
            temporal_attention = torch.einsum('bif,bnf->bin', torch.stack([central_nodes_time_embedding for _ in range(self.num_items)], dim=1), hop_nodes_time_embedding)

            temporal_attention = self.temporal_attention_dropout(temporal_attention)

            attention = attention + self.temporal_information_importance * temporal_attention

            attention = attention + attention_mask

            attention = self.leaky_relu_func(attention)

            # shape (batch_size, num_items, max_neighbors_num)
            attention_scores = F.softmax(attention, dim=-1)

            # shape (batch_size, num_items, embedding_dimension),  (batch_size, num_items, max_neighbors_num) bmm (batch_size, max_neighbors_num, embedding_dimension)
            hop_embedding = torch.bmm(attention_scores, hop_nodes_embedding)

            nodes_hops_embedding.append(hop_embedding)

        # (batch_size, num_items, hop_num, embedding_dimension)
        nodes_hops_embedding = self.embedding_dropout(torch.stack(nodes_hops_embedding, dim=2))

        # make final prediction with concatenation operation
        # nodes_embedding_projection, shape (batch_size, num_items, embedding_dimension)
        nodes_embedding_projection = self.fc_projection(nodes_hops_embedding.flatten(start_dim=2))
        # set_prediction, shape (batch_size, num_items),   (batch_size, num_items, embedding_dimension) * (num_items, embedding_dimension)
        set_prediction = (nodes_embedding_projection * query_embeddings).sum(dim=-1)

        return set_prediction
