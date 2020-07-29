"""
    Utility functions for training one epoch
    and evaluating one epoch
"""
import torch
import torch.nn as nn
import math

from train.metrics import accuracy_TU as accuracy
from sklearn.metrics import balanced_accuracy_score

def train_epoch(model, optimizer, device, data_loader, epoch):
    model.train()
    epoch_loss = 0
    epoch_train_acc = 0
    nb_data = 0
    gpu_mem = 0
    total_labs = []
    total_scor = []
    for iter, (batch_graphs, batch_labels, batch_snorm_n, batch_snorm_e) in enumerate(data_loader):
        batch_x = batch_graphs.ndata['feat'].to(device)  # num x feat
        batch_e = batch_graphs.edata['feat'].to(device)
        batch_snorm_e = batch_snorm_e.to(device)
        batch_labels = batch_labels.to(device)
        batch_snorm_n = batch_snorm_n.to(device)         # num x 1
        optimizer.zero_grad()
        batch_scores = model.forward(batch_graphs, batch_x, batch_e, batch_snorm_n, batch_snorm_e)
        loss = model.loss(batch_scores, batch_labels)
        loss.backward()
        optimizer.step()
        epoch_loss += loss.detach().item()
        total_labs.extend(batch_labels.tolist())
        total_scor.extend(batch_scores.detach().argmax(dim=1).tolist())
        # epoch_train_acc += accuracy(batch_labels, batch_scores)
        nb_data += batch_labels.size(0)
    epoch_loss /= (iter + 1)
    # epoch_train_acc /= nb_data
    # epoch_train_acc /= (iter + 1)

    return epoch_loss, balanced_accuracy_score(total_labs,total_scor), optimizer

def evaluate_network(model, device, data_loader, epoch):
    model.eval()
    epoch_test_loss = 0
    epoch_test_acc = 0
    nb_data = 0
    total_labs = []
    total_scor = []
    with torch.no_grad():
        for iter, (batch_graphs, batch_labels, batch_snorm_n, batch_snorm_e) in enumerate(data_loader):
            batch_x = batch_graphs.ndata['feat'].to(device)
            batch_e = batch_graphs.edata['feat'].to(device)
            batch_snorm_e = batch_snorm_e.to(device)
            batch_labels = batch_labels.to(device)
            batch_snorm_n = batch_snorm_n.to(device)

            batch_scores = model.forward(batch_graphs, batch_x, batch_e, batch_snorm_n, batch_snorm_e)
            loss = model.loss(batch_scores, batch_labels)
            epoch_test_loss += loss.detach().item()
            total_labs.extend(batch_labels.tolist())
            total_scor.extend(batch_scores.detach().argmax(dim=1).tolist())
            # epoch_test_acc += accuracy(batch_labels, batch_scores)
            nb_data += batch_labels.size(0)
        epoch_test_loss /= (iter + 1)
        # epoch_test_acc /= nb_data
        # epoch_test_acc /= (iter + 1)

    return epoch_test_loss, balanced_accuracy_score(total_labs,total_scor)

def evaluate_network2(model, device, data_loader, epoch):
    model.eval()
    epoch_test_loss = 0
    epoch_test_acc = 0
    nb_data = 0
    total_labs = []
    total_scor = []
    with torch.no_grad():
        for iter, (batch_graphs, batch_labels, batch_snorm_n, batch_snorm_e) in enumerate(data_loader):
            batch_x = batch_graphs.ndata['feat'].to(device)
            batch_e = batch_graphs.edata['feat'].to(device)
            batch_snorm_e = batch_snorm_e.to(device)
            batch_labels = batch_labels.to(device)
            batch_snorm_n = batch_snorm_n.to(device)

            batch_scores = model.forward(batch_graphs, batch_x, batch_e, batch_snorm_n, batch_snorm_e)
            loss = model.loss(batch_scores, batch_labels)
            epoch_test_loss += loss.detach().item()
            total_labs.extend(batch_labels.tolist())
            total_scor.extend(batch_scores.detach().tolist())
            # epoch_test_acc += accuracy(batch_labels, batch_scores)
            nb_data += batch_labels.size(0)
        epoch_test_loss /= (iter + 1)
        # epoch_test_acc /= nb_data
        # epoch_test_acc /= (iter + 1)

    return epoch_test_loss, total_labs, total_scor
    
def check_patience(all_losses, best_loss, best_epoch, curr_loss, curr_epoch, counter):
    if curr_loss < best_loss:
        counter = 0
        best_loss = curr_loss
        best_epoch = curr_epoch
    else:
        counter += 1
    return best_loss, best_epoch, counter
