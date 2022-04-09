import torch
import torch.nn as nn
import os
import shutil
import json
from tqdm import tqdm
import warnings
import numpy as np
import logging
import time

from model.ETGNN import ETGNN
from utils.utils import get_optimizer, get_lr_scheduler
from utils.metric import evaluate_model, get_metrics
from utils.load_config import get_attribute, config
from utils.utils import convert_to_gpu, get_n_params, set_random_seed
from utils.EarlyStopping import EarlyStopping
from utils.DataLoaders import get_node_idx_data_loader, CustomizedDataLoader

if __name__ == '__main__':
    warnings.filterwarnings('ignore')

    # set up logger
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger()
    logger.setLevel(logging.DEBUG)
    os.makedirs(f"../logs/{get_attribute('dataset_name')}/{get_attribute('model_name')}", exist_ok=True)
    # create file handler that logs debug and higher level messages
    fh = logging.FileHandler(
        f"../logs/{get_attribute('dataset_name')}/{get_attribute('model_name')}/{str(time.time())}.log")
    fh.setLevel(logging.DEBUG)
    # create console handler with a higher log level
    ch = logging.StreamHandler()
    ch.setLevel(logging.WARNING)
    # create formatter and add it to the handlers
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    fh.setFormatter(formatter)
    ch.setFormatter(formatter)
    # add the handlers to logger
    logger.addHandler(fh)
    logger.addHandler(ch)

    set_random_seed(get_attribute('seed'))

    logger.info('getting data loaders...')
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

    logger.info('creating model...')
    model = ETGNN(num_items=get_attribute('num_items'), num_users=get_attribute('num_users'),
                  hop_num=get_attribute('hop_num'), embedding_dimension=get_attribute('embedding_dimension'),
                  temporal_feature_dimension=train_data_loader.temporal_feature_dimension,
                  embedding_dropout=get_attribute('embedding_dropout'),
                  temporal_attention_dropout=get_attribute('temporal_attention_dropout'),
                  temporal_information_importance=get_attribute('temporal_feature_importance'))

    model = convert_to_gpu(model, device=get_attribute('device'))

    logger.info(model)

    logger.info(f'Model #Params: {get_n_params(model) * 4} B, {get_n_params(model) * 4 / 1024} KB, {get_n_params(model) * 4 / 1024 / 1024} MB.')

    optimizer = get_optimizer(model, get_attribute('optimizer'), get_attribute('learning_rate'), get_attribute('weight_decay'))

    scheduler = get_lr_scheduler(optimizer, learning_rate=get_attribute('learning_rate'), t_max=5000)

    save_model_folder = f"../save_model_folder/{get_attribute('dataset_name')}/{get_attribute('model_name')}"
    shutil.rmtree(save_model_folder, ignore_errors=True)
    os.makedirs(save_model_folder, exist_ok=True)

    early_stopping = EarlyStopping(patience=get_attribute('patience'), save_model_folder=save_model_folder,
                                   save_model_name=get_attribute('model_name'), logger=logger)

    loss_func = nn.MultiLabelSoftMarginLoss(reduction="mean")

    logger.info(f'configuration is {config}')

    data_loader_dic = {"train": train_data_loader, "validate": valid_data_loader, "test": test_data_loader}

    for epoch in range(get_attribute('epochs')):
        loss_dict, metric_dict = {}, {}

        for mode in ["train", "validate", "test"]:
            # training
            if mode == "train":
                model.train()
            # validate or test
            else:
                model.eval()

            y_true = []
            y_pred = []
            total_loss = 0.0
            tqdm_loader = tqdm(node_idx_data_loader, ncols=120)
            for batch, user_indices in enumerate(tqdm_loader):
                # user_indices -> list of batch_size user indices (str format)
                # hops_nodes_indices, list, shape (1 + hop_num, batch_size, max_neighbors_num), each element is a Tensor
                # hops_nodes_temporal_features, list, shape, (1 + hop_num, batch_size, max_neighbors_num, temporal_feature_dimension), each element is a Tensor
                # hops_nodes_length, list, shape (1 + hop_num, batch_size)
                # central_nodes_temporal_feature, Tensor, shape (batch_size, temporal_feature_dimension)
                # central_nodes_label, Tensor, shape (batch_size, num_items)
                hops_nodes_indices, hops_nodes_temporal_features, hops_nodes_length, central_nodes_temporal_feature, central_nodes_label = \
                    data_loader_dic[mode].get_batch_nodes_data(user_indices)

                for hop_index in range(len(hops_nodes_indices)):
                    hops_nodes_indices[hop_index] = convert_to_gpu(hops_nodes_indices[hop_index], device=get_attribute('device'))
                    hops_nodes_temporal_features[hop_index] = convert_to_gpu(hops_nodes_temporal_features[hop_index], device=get_attribute('device'))

                central_nodes_temporal_feature, truth_data = convert_to_gpu(central_nodes_temporal_feature, central_nodes_label, device=get_attribute('device'))

                with torch.set_grad_enabled(mode == 'train'):
                    # (batch_size, num_items)
                    output = model(hops_nodes_indices, hops_nodes_temporal_features, hops_nodes_length, central_nodes_temporal_feature)
                    loss = loss_func(output, truth_data)
                    total_loss += loss.item()
                    if mode == "train":
                        optimizer.zero_grad()
                        loss.backward()
                        optimizer.step()
                        scheduler.step()

                    y_pred.append(output.detach().cpu())
                    y_true.append(truth_data.detach().cpu())
                    tqdm_loader.set_description(f'{mode} epoch: {epoch + 1}, batch: {batch + 1}, {mode} loss: {total_loss / (batch + 1)}')

            loss_dict[mode] = total_loss / (batch + 1)
            y_true = torch.cat(y_true, dim=0)
            y_pred = torch.cat(y_pred, dim=0)

            metrics = get_metrics(y_true=y_true, y_pred=y_pred)
            metric_dict[mode] = metrics

            logger.info(f'Epoch: {epoch + 1}, {mode} metrics: {metrics}')

        # save best model using validate data
        validate_ndcg = np.mean([metric_dict["validate"][key] for key in metric_dict["validate"] if key.startswith(f"ndcg_")])
        early_stop = early_stopping.step([('ndcg', validate_ndcg, True)], model)

        if early_stop:
            break

    # load best model
    early_stopping.load_checkpoint(model)

    logger.info('getting final performance...')

    train_metrics = evaluate_model(model, node_idx_data_loader, data_loader_dic['train'], device=get_attribute('device'))
    valid_metrics = evaluate_model(model, node_idx_data_loader, data_loader_dic['validate'], device=get_attribute('device'))
    test_metrics = evaluate_model(model, node_idx_data_loader, data_loader_dic['test'], device=get_attribute('device'))

    logger.info(f'train metrics -> {train_metrics}')
    logger.info(f'validate metrics -> {valid_metrics}')
    logger.info(f'test metrics -> {test_metrics}')

    save_result_folder = f"../results/{get_attribute('dataset_name')}"
    if not os.path.exists(save_result_folder):
        os.makedirs(save_result_folder, exist_ok=True)

    save_result_path = f"{save_result_folder}/{get_attribute('model_name')}.json"
    with open(save_result_path, 'w') as file:
        metrics_str = json.dumps({"train": train_metrics, "validate": valid_metrics, "test": test_metrics}, indent=4)
        file.write(metrics_str)
        logger.info(f'result saves at {save_result_path} successfully.')
