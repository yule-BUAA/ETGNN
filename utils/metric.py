import torch
from tqdm import tqdm
import torch.nn as nn
from utils.utils import convert_to_gpu
from torch.utils.data import DataLoader
from utils.DataLoaders import CustomizedDataLoader


def recall_score(y_true: torch.Tensor, y_pred: torch.Tensor, top_k: int = 10):
    """
    Args:
        y_true (Tensor): shape (batch_size, num_items)
        y_pred (Tensor): shape (batch_size, num_items)
        top_k (int):
    Returns:
        output (float)
    """
    _, predict_indices = y_pred.topk(k=top_k)
    predict, truth = y_pred.new_zeros(y_pred.shape).scatter_(dim=1, index=predict_indices,
                                                             value=1).long(), y_true.long()
    tp, t = ((predict == truth) & (truth == 1)).sum(-1), truth.sum(-1)
    return (tp.float() / t.float()).mean().item()


def dcg(y_true: torch.Tensor, y_pred: torch.Tensor, top_k: int = 10):
    """
    Args:
        y_true: (batch_size, num_items)
        y_pred: (batch_size, num_items)
        top_k (int):

    Returns:

    """
    _, predict_indices = y_pred.topk(k=top_k)
    gain = y_true.gather(-1, predict_indices)  # (batch_size, top_k)
    return (gain.float() / torch.log2(torch.arange(top_k, device=y_pred.device).float() + 2)).sum(-1)  # (batch_size,)


def ndcg_score(y_true: torch.Tensor, y_pred: torch.Tensor, top_k: int = 10):
    """
    Args:
        y_true: (batch_size, num_items)
        y_pred: (batch_size, num_items)
        top_k (int):
    Returns:

    """
    dcg_score = dcg(y_true, y_pred, top_k)
    idcg_score = dcg(y_true, y_true, top_k)
    return (dcg_score / idcg_score).mean().item()


def PHR(y_true: torch.Tensor, y_pred: torch.Tensor, top_k: int = 10):
    """
    Args:
        y_true (Tensor): shape (batch_size, num_items)
        y_pred (Tensor): shape (batch_size, num_items)
        top_k (int):
    Returns:
        output (float)
    """
    _, predict_indices = y_pred.topk(k=top_k)
    predict, truth = y_pred.new_zeros(y_pred.shape).scatter_(dim=1, index=predict_indices,
                                                             value=1).long(), y_true.long()

    hit_num = torch.mul(predict, truth).sum(dim=1).nonzero().shape[0]
    return hit_num / truth.shape[0]


def get_metrics(y_true: torch.Tensor, y_pred: torch.Tensor):
    """
        Args:
            y_true: tensor (samples_num, num_items)
            y_pred: tensor (samples_num, num_items)
        Returns:
            scores: dict
    """
    result = {}
    for top_k in [10, 20, 30, 40]:
        result.update({
            f'recall_{top_k}': recall_score(y_true, y_pred, top_k=top_k),
            f'ndcg_{top_k}': ndcg_score(y_true, y_pred, top_k=top_k),
            f'PHR_{top_k}': PHR(y_true, y_pred, top_k=top_k)
        })
    result = sorted(result.items(), key=lambda item: item[0], reverse=False)
    result = {item[0]: float(f"{item[1]:.4f}") for item in result}
    return result


def evaluate_model(model: nn.Module, node_idx_data_loader: DataLoader, data_loader: CustomizedDataLoader, device: str):
    """
    evaluate model, return the metrics
    Args:
        model: nn.Module
        node_idx_data_loader: DataLoader for node indices
        data_loader: DataLoader for nodes data
        device: gpu device
    Returns:
        metrics: dict
    """
    model.eval()

    y_true = []
    y_pred = []

    with torch.no_grad():
        tqdm_loader = tqdm(node_idx_data_loader, ncols=120)
        for batch, user_indices in enumerate(tqdm_loader):
            # user_indices -> list of batch_size user indices (str format)
            # hops_nodes_indices, list, shape (1 + hop_num, batch_size, max_neighbors_num), each element is a Tensor
            # hops_nodes_temporal_features, list, shape, (1 + hop_num, batch_size, max_neighbors_num, temporal_feature_dimension), each element is a Tensor
            # hops_nodes_length, list, shape (1 + hop_num, batch_size)
            # central_nodes_temporal_feature, Tensor, shape (batch_size, temporal_feature_dimension)
            # central_nodes_label, Tensor, shape (batch_size, num_items)
            hops_nodes_indices, hops_nodes_temporal_features, hops_nodes_length, central_nodes_temporal_feature, central_nodes_label = \
                data_loader.get_batch_nodes_data(user_indices)

            for hop_index in range(len(hops_nodes_indices)):
                hops_nodes_indices[hop_index] = convert_to_gpu(hops_nodes_indices[hop_index], device=device)
                hops_nodes_temporal_features[hop_index] = convert_to_gpu(hops_nodes_temporal_features[hop_index], device=device)

            central_nodes_temporal_feature, truth_data = convert_to_gpu(central_nodes_temporal_feature, central_nodes_label, device=device)

            # (batch_size, num_items)
            output = model(hops_nodes_indices, hops_nodes_temporal_features, hops_nodes_length, central_nodes_temporal_feature)

            y_pred.append(output.detach().cpu())
            y_true.append(truth_data.detach().cpu())
            tqdm_loader.set_description(f'batch: {batch + 1}')

        y_true = torch.cat(y_true, dim=0)
        y_pred = torch.cat(y_pred, dim=0)

    return get_metrics(y_true=y_true, y_pred=y_pred)
