import os
import random
import time
#import wandb
import torch
import numpy as np

from models.model_hyper_cat import Model
from utils import load_adj, EHRDataset, format_time, MultiStepLRScheduler, med_load_adj
from metrics_med import evaluate_codes, evaluate_hf
from config import sweep_config, hyperparameter_defaults


def historical_hot(code_x, code_num, lens):
    result = np.zeros((len(code_x), code_num), dtype=int)
    for i, (x, l) in enumerate(zip(code_x, lens)):
        result[i] = x[l - 1]
    return result


if __name__ == '__main__':
#def train():
    #wandb.init(config=hyperparameter_defaults)
    #config = wandb.config
    seed = 6699
    dataset = 'mimic3'#'mimic3'  # 'mimic3' or 'eicu'
    task = 'm'#'m'  # 'm' or 'h'
    use_cuda = True
    device = torch.device('cuda' if torch.cuda.is_available() and use_cuda else 'cpu')

    code_size = 256#48
    graph_size = 32#32#32
    hidden_size = 256#200#150  # rnn hidden size #输出的维度
    t_attention_size = 256#150
    t_output_size = hidden_size
    batch_size = 32
    epochs = 200

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)

    dataset_path = os.path.join('data', dataset, 'standard')
    train_path = os.path.join(dataset_path, 'train')
    valid_path = os.path.join(dataset_path, 'valid')
    test_path = os.path.join(dataset_path, 'test')

    code_adj = load_adj(dataset_path, device=device)
    med_code_adj = med_load_adj(dataset_path, device=device)
    code_num = len(code_adj)
    med_code_num = len(med_code_adj)
    print('loading train data ...')
    train_data = EHRDataset(train_path, label=task, batch_size=batch_size, shuffle=True, device=device)
    print('loading valid data ...')
    valid_data = EHRDataset(valid_path, label=task, batch_size=batch_size, shuffle=False, device=device)
    print('loading test data ...')
    test_data = EHRDataset(test_path, label=task, batch_size=batch_size, shuffle=False, device=device)

    test_historical = historical_hot(valid_data.code_x, code_num, valid_data.visit_lens)

    task_conf = {
        'm': {
            'dropout': 0.45,
            'output_size': code_num,
            'evaluate_fn': evaluate_codes,
            'lr': {
                'init_lr': 0.01,
                'milestones': [20, 30],
                'lrs': [1e-3, 1e-5]
            }
        },
        'h': {
            'dropout': 0.0,
            'output_size': 1,
            'evaluate_fn': evaluate_hf,
            'lr': {
                'init_lr': 0.01,
                'milestones': [2, 3, 20],
                'lrs': [1e-3, 1e-4, 1e-5]
            }
        }
    }
    output_size = task_conf[task]['output_size']
    activation = torch.nn.Sigmoid()
    loss_fn = torch.nn.BCELoss()
    evaluate_fn = task_conf[task]['evaluate_fn']
    dropout_rate = task_conf[task]['dropout']

    param_path = os.path.join('data', 'params', dataset, task)
    if not os.path.exists(param_path):
        os.makedirs(param_path)

    model = Model(code_num=code_num, code_size=code_size,
                  adj=code_adj, graph_size=graph_size, hidden_size=hidden_size, trans_embedding_dim=hidden_size, t_attention_size=t_attention_size,
                  t_output_size=t_output_size,
                  output_size=output_size, dropout_rate=dropout_rate, activation=activation, med_code_num = med_code_num).to(device)
    #wandb.watch(model)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    scheduler = MultiStepLRScheduler(optimizer, epochs, task_conf[task]['lr']['init_lr'],
                                     task_conf[task]['lr']['milestones'], task_conf[task]['lr']['lrs'])

    pytorch_total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    #print(pytorch_total_params)
    best_f1 = 0
    for epoch in range(epochs):
        print('Epoch %d / %d:' % (epoch + 1, epochs))
        model.train()
        total_loss = 0.0
        total_num = 0
        steps = len(train_data)
        st = time.time()
        scheduler.step()
        for step in range(len(train_data)):
            optimizer.zero_grad()
            code_x, visit_lens, divided, y, neighbors, medicine_codes = train_data[step]
            output = model(code_x, divided, neighbors, visit_lens, medicine_codes).squeeze()
            loss = loss_fn(output, y)
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * output_size * len(code_x)
            total_num += len(code_x)
            end_time = time.time()
            remaining_time = format_time((end_time - st) / (step + 1) * (steps - step - 1))
            print('\r    Step %d / %d, remaining time: %s, loss: %.4f'
                  % (step + 1, steps, remaining_time, total_loss / total_num), end='')
        train_data.on_epoch_end()
        et = time.time()
        time_cost = format_time(et - st)

        print('\r    Step %d / %d, time cost: %s, loss: %.4f' % (steps, steps, time_cost, total_loss / total_num))
        valid_loss, f1_score, recall = evaluate_fn(model, valid_data, loss_fn, output_size, test_historical)
        #metrics = {'f1_score': f1_score, 'loss': valid_loss, 'recall_10': recall[0], 'recall_20': recall[1]}
        #wandb.log(metrics)
        if f1_score > best_f1:
            best_f1 = f1_score
            # torch.save(model.state_dict(), os.path.join(param_path, '%d.pt' % epoch))
        print('best_f1=%d', best_f1)
