from evaluations import train_and_test_models
import torch.nn as nn
import torch.optim as optim
import time
import torch
import os
from torch.utils.data import DataLoader
import numpy as np
import logging

logger = logging.getLogger(__name__)

def count_parameters(model, trainable=False):
    if trainable:
        return sum(p.numel() for p in model.parameters() if p.requires_grad)
    return sum(p.numel() for p in model.parameters())

def train_model_generic(model, train_loader, test_loader, device, learning_rate=1e-3, weight_decay=1e-4, gamma=0.8, epochs= 15, path = "drive/MyDrive/KANs/models"):
    model.to(device)
    logger.info(f"Training model {model.name}")
    logger.info(f"Total parameters: {count_parameters(model, trainable=False)}")
    logger.info(f"Trainable parameters: {count_parameters(model, trainable=True)}")

    optimizer = optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    scheduler = optim.lr_scheduler.ExponentialLR(optimizer, gamma=gamma)
    criterion = nn.CrossEntropyLoss()
    logger.info(f"Optimizer: AdamW, LR: {learning_rate}, Scheduler: ExponentialLR (gamma={gamma})")
    start = time.perf_counter()
    # DataLoader
    all_train_loss, all_test_loss, all_test_accuracy, all_test_precision, all_test_recall, all_test_f1, all_learning_rates, all_inference_time = train_and_test_models(model, device, train_loader, test_loader, optimizer, criterion, epochs=epochs, scheduler=scheduler, path= path, save_last=True)
    total_time = time.perf_counter() - start
    avg_epoch_time_per_min = (total_time / 60)/epochs
    model.training_time = avg_epoch_time_per_min
    final_inference_time_ms = all_inference_time[-1] if all_inference_time else None
    if not path is None:
        saving_path = os.path.join(path, model.name + "_last.pt")
        model =  torch.load(saving_path, map_location=torch.device(device), weights_only=False)
        model.train_losses = all_train_loss
        model.test_losses = all_test_loss
        torch.save(model, saving_path)
    logger.info("--- Training Summary ---")
    logger.info(f"Train loss: {all_train_loss}")
    logger.info(f"Test loss: {all_test_loss}")
    logger.info(f"All test accuracy: {all_test_accuracy}")
    logger.info(f"All test precision: {all_test_precision}")
    logger.info(f"All test recall: {all_test_recall}")
    logger.info(f"All test f1: {all_test_f1}")
    logger.info(f"All test learning rates: {all_learning_rates}")
    logger.info(f"Total training time: {total_time/60:.2f} minutes")
    logger.info(f"Average time per epoch: {avg_epoch_time_per_min:.2f} minutes")
    if final_inference_time_ms is not None:
        logger.info(f"Final average inference time: {final_inference_time_ms:.4f} ms/image")
    logger.info("-"*200)
    #return all_train_loss, all_test_loss, all_test_accuracy, all_test_precision, all_test_recall, all_test_f1