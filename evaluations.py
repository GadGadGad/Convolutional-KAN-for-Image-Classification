import time
from tqdm import tqdm
import torch
from sklearn.metrics import precision_score, recall_score, f1_score
import os
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import RocCurveDisplay
import logging
import torch.nn.functional as F

logger = logging.getLogger(__name__)

def train(model, device, train_loader, optimizer, epoch_num, criterion):
    """
    Train the model for one epoch

    Args:
        model: the neural network model
        device: cuda or cpu
        train_loader: DataLoader for training data
        optimizer: the optimizer to use (e.g. SGD)
        epoch: the current epoch
        criterion: the loss function (e.g. CrossEntropy)

    Returns:
        avg_loss: the average loss over the training set
    """

    model.to(device)
    model.train()
    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0) # Prevent exploding gradients
    train_loss = 0
    total_train_time = 0.0
    if device.type == 'cuda':
        start_event = torch.cuda.Event(enable_timing=True)
        end_event = torch.cuda.Event(enable_timing=True)

    progress_bar = tqdm(train_loader, desc=f"Training epoch {epoch_num}", leave=False)
    for batch_idx, (data, target) in enumerate(progress_bar):
        data, target = data.to(device), target.to(device)
        # Reset the optimizer
        optimizer.zero_grad()

        # Push the data forward through the model layers
        if device.type == 'cuda':
            start_event.record()
        else:
            start_time = time.perf_counter()
        output = model(data)
        if device.type == 'cuda':
            end_event.record()
            torch.cuda.synchronize()
            batch_train_time = start_event.elapsed_time(end_event) / 1000.0
        else:
            end_time = time.perf_counter()
            batch_train_time = end_time - start_time
        total_train_time += batch_train_time
        # Get the loss
        loss = criterion(output, target)
        # Keep a running total for the final average
        train_loss += loss.item()
        # Backpropagate
        loss.backward()
        # for name, param in model.named_parameters():
        #     if param.grad is None:
        #         print(f"NO GRAD: {name}")
        #     else:
        #         print(f"GRAD NORM {name}: {param.grad.norm().item()}")
        #         pass 
        optimizer.step()

        current_avg_loss = train_loss / (batch_idx + 1)
        progress_bar.set_postfix(loss=f'{current_avg_loss:.4f}')
    # Final average loss for the entire epoch
    avg_loss = train_loss / (batch_idx + 1)

    return avg_loss, total_train_time

def test(model, device, test_loader, epoch_num, criterion):
    """
    Test the model

    Args:
        model: the neural network model
        device: cuda or cpu
        test_loader: DataLoader for test data
        criterion: the loss function (e.g. CrossEntropy)

    Returns:
        test_loss: the average loss over the test set (normalized by dataset size)
        accuracy: the accuracy of the model on the test set
        precision: the precision of the model on the test set (macro avg)
        recall: the recall of the model on the test set (macro avg)
        f1: the f1 score of the model on the test set (macro avg)
    """

    model.eval()
    test_loss = 0
    correct = 0
    all_targets = []
    all_predictions = []
    total_inference_time = 0.0
    test_loss_sum = 0.0 
    num_images = 0
    if device.type == 'cuda':
       start_event = torch.cuda.Event(enable_timing=True)
       end_event = torch.cuda.Event(enable_timing=True)

    with torch.no_grad():
        # Create tqdm iterator manually for consistency
        progress_bar = tqdm(test_loader, desc=f"Testing epoch {epoch_num}", leave=False)
        for batch_idx, (data, target) in enumerate(progress_bar):
            data, target = data.to(device), target.to(device)
            batch_size = data.shape[0]
            num_images += batch_size
            if device.type == 'cuda':
                start_event.record()
            else:
                start_time = time.perf_counter()
            output = model(data)
            if device.type == 'cuda':
                end_event.record()
                torch.cuda.synchronize()
                batch_inference_time = start_event.elapsed_time(end_event) / 1000.0
            else:
                end_time = time.perf_counter()
                batch_inference_time = end_time - start_time
            total_inference_time += batch_inference_time
            batch_loss = criterion(output, target).item()
            test_loss_sum += batch_loss

            # Calculate the accuracy for this batch
            _, predicted = torch.max(output.data, 1)
            correct += (target == predicted).sum().item()

            # Collect all targets and predictions for metric calculations
            all_targets.extend(target.view_as(predicted).cpu().numpy())
            all_predictions.extend(predicted.cpu().numpy())

            current_avg_loss = test_loss_sum / (batch_idx + 1)
            progress_bar.set_postfix(loss=f'{current_avg_loss:.4f}')


    precision = precision_score(all_targets, all_predictions, average='macro', zero_division=0)
    recall = recall_score(all_targets, all_predictions, average='macro', zero_division=0)
    f1 = f1_score(all_targets, all_predictions, average='macro', zero_division=0)

    test_loss = test_loss_sum / len(test_loader)
    accuracy = correct / len(test_loader.dataset)
    avg_inference_time = (total_inference_time / num_images) * 1000
    return test_loss, accuracy, precision, recall, f1, avg_inference_time


def train_and_test_models(model, device, train_loader, test_loader, optimizer, criterion, epochs, scheduler, path = "drive/MyDrive/KANs/models", verbose = True, save_last=False, patience = np.inf):
    """
    Train and test the model

    Args:
        model: the neural network model
        device: cuda or cpu
        train_loader: DataLoader for training data
        test_loader: DataLoader for test data
        optimizer: the optimizer to use (e.g. SGD)
        criterion: the loss function (e.g. CrossEntropy)
        epochs: the number of epochs to train
        scheduler: the learning rate scheduler

    Returns:
        all_train_loss: a list of the average training loss for each epoch
        all_test_loss: a list of the average test loss for each epoch
        all_test_accuracy: a list of the accuracy for each epoch
        all_test_precision: a list of the precision for each epoch
        all_test_recall: a list of the recall for each epoch
        all_test_f1: a list of the f1 score for each epoch
        all_leraning_rates: a list of the learning rate for each epoch
    """
    # Track metrics
    all_inference_time = []
    all_train_loss = []
    all_test_loss = []
    all_test_accuracy = []
    all_test_precision = []
    all_test_recall = []
    all_test_f1 = []
    all_learning_rates = []
    best_acc = 0
    havent_improved = 0
    best_epoch = 0
    # Create results directory if it doesn't exist
    os.makedirs(path, exist_ok=True)

    for epoch in range(1, epochs + 1):
        current_lr = optimizer.param_groups[0]['lr']
        all_learning_rates.append(current_lr)
        logger.info(f"--- Starting Epoch {epoch}/{epochs} --- LR: {current_lr:.4e} ---")
        # Train the model
        train_loss, train_time = train(model, device, train_loader, optimizer, epoch, criterion)
        all_train_loss.append(train_loss)
        # Test the model
        test_loss, test_accuracy, test_precision, test_recall, test_f1, avg_inference_time = test(model, device, test_loader, epoch, criterion)
        all_test_loss.append(test_loss)
        all_test_accuracy.append(test_accuracy)
        all_test_precision.append(test_precision)
        all_test_recall.append(test_recall)
        all_test_f1.append(test_f1)
        all_inference_time.append(avg_inference_time)
        # Note: train_loss and test_loss here are the *average* losses for the epoch
        log_msg = (
            f"Epoch {epoch}/{epochs} Summary: "
            f"\nAvg Train Loss: {train_loss:.6f}, Avg Test Loss: {test_loss:.6f}, "
            f"\nAccuracy: {test_accuracy:.4f}, Precision: {test_precision:.4f}, "
            f"\nRecall: {test_recall:.4f}, F1: {test_f1:.4f}, "
            f"\nTraining Time: {train_time:.4f}"
            f"\nAvg Inference Time: {avg_inference_time:.4f} ms/image"
        )

        logger.info(log_msg)
        if test_accuracy>best_acc:
            havent_improved = 0
            best_acc = test_accuracy
            best_epoch = epoch
            logger.info(f"Found new best accuracy: {best_acc:.4f} at epoch {epoch}")
            torch.save(model,os.path.join(path,model.name+"_best.pt"))
        else: havent_improved+=1

        if not (scheduler is None):
            scheduler.step()
        if havent_improved>patience: # early stopping
            logger.info(f"Early stopping triggered after {patience} epochs without improvement.")
            break
    model.all_test_accuracy = all_test_accuracy
    model.all_test_precision = all_test_precision
    model.all_test_f1 = all_test_f1
    model.all_test_recall = all_test_recall
    model.train_losses = all_train_loss
    model.test_losses = all_test_loss

    logger.info(f"Training finished. Best test accuracy: {best_acc:.4f} achieved at epoch {best_epoch}")
    if verbose:
        print("Best test accuracy", best_acc)
    if save_last:
        torch.save(model,os.path.join(path, model.name + "_last.pt"))


    return all_train_loss, all_test_loss, all_test_accuracy, all_test_precision, all_test_recall, all_test_f1, all_learning_rates, all_inference_time

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def highlight_max(s):
    is_numeric = pd.to_numeric(s, errors='coerce').notna()
    if not is_numeric.all(): 
        return ['' for _ in s]
    s_numeric = s[is_numeric].astype(float)
    if s_numeric.empty:
        return ['' for _ in s]
    max_val = s_numeric.max()
    return ['font-weight: bold' if is_numeric[i] and v == max_val else '' for i, v in enumerate(s)]


def final_plots(models, test_loader, criterion, device, use_time = False):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 5))
    accs = []
    precisions = []
    recalls = []
    f1s = []
    params_counts = []
    times = []
    model_names = []
    test_losses = []

    for model in models:
        # Ensure model is on the correct device for testing
        model.to(device)
        # Re-run test to get final metrics after training is complete
        test_loss, accuracy, precision, recall, f1 = test(model, device, test_loader, 'final', criterion)
        model_name = getattr(model, 'name', f'Model_{len(model_names)}') # Get name or assign default
        model_names.append(model_name)
        test_losses.append(test_loss)
        # Check if model has test_losses attribute from training history
        if hasattr(model, 'test_losses') and model.test_losses:
            ax1.plot(model.test_losses, label=model_name)
        else:
            logger.warning(f"Model {model_name} has no 'test_losses' attribute to plot.")

        params = count_parameters(model)
        ax2.scatter(params, accuracy, label=model_name)
        accs.append(accuracy)
        precisions.append(precision)
        recalls.append(recall)
        f1s.append(f1)
        params_counts.append(params)
        if use_time and hasattr(model, 'training_time'):
            times.append(model.training_time)
        else:
            times.append(np.nan) # Use NaN if time is not used or unavailable

    ax1.set_title('Test Loss vs Epochs (From Training History)')
    ax1.set_xlabel('Epochs')
    ax1.set_ylabel('Loss')
    if any(hasattr(m, 'test_losses') and m.test_losses for m in models):
        ax1.legend() # Only show legend if there's something to label
    ax1.grid(True)

    ax2.set_title('Number of Parameters vs Final Accuracy')
    ax2.set_xlabel('Number of Parameters')
    ax2.set_ylabel('Accuracy')
    ax2.legend()
    ax2.grid(True)
    ax2.ticklabel_format(style='sci', axis='x', scilimits=(0,0))

    plt.tight_layout()
    plt.show()

    df_data = {
        "Final Test Accuracy": accs,
        "Final Test Precision (Macro)": precisions,
        "Final Test Recall (Macro)": recalls,
        "Final Test F1 Score (Macro)": f1s,
        "Number of Parameters": params_counts,
    }
    if use_time:
        df_data["Avg Epoch Time (min)"] = times

    df = pd.DataFrame(df_data, index=model_names)

    df.to_csv('experiment_results.csv', index=True, index_label='Model Name')
    logger.info("Saved experiment results to experiment_results.csv")


    valid_cols = [col for col in df.columns if col in df_data]
    df_styled = df.style.apply(highlight_max, subset=valid_cols, axis=0).format('{:.4f}', na_rep='N/A')
    if "Number of Parameters" in df.columns:
         df_styled = df_styled.format({"Number of Parameters": '{:,}'})
    return df_styled


def plot_roc_one_vs_rest_all_models(models, dataloader, n_classes, device, class_names=None):
    num_models = len(models)
    if num_models == 0:
        logger.warning("No models provided for ROC plotting.")
        return

    fig, axs = plt.subplots(num_models, 1, figsize=(7, 6 * num_models), squeeze=False) # Ensure axs is always 2D

    for i, model in enumerate(models):
        model_name = getattr(model, 'name', f'Model {i+1}')
        plot_roc_one_vs_rest(model, dataloader, n_classes, device, axs[i, 0], model_name, class_names) # Pass model_name

    plt.tight_layout()
    plt.show()


def plot_roc_one_vs_rest(model, dataloader, n_classes, device, ax, model_name="Model", class_names=None):
    """ Plots the One-vs-Rest ROC curves for a given model. """

    with torch.no_grad():
        all_probs = []
        all_targets = []
        model.eval()
        model.to(device) # Ensure model is on the right device

        for data, target in tqdm(dataloader, desc=f"ROC Calc ({model_name})", leave=False):
            data, target = data.to(device), target.to(device)
            output = model(data)
            # Convert logits to probabilities using softmax
            probs = F.softmax(output, dim=1)
            all_probs.append(probs.cpu().numpy())
            all_targets.append(target.cpu().numpy())

    predictions = np.concatenate(all_probs)
    targets = np.concatenate(all_targets)

    if class_names is None:
        class_names = [f"Class {i}" for i in range(n_classes)]
    elif len(class_names) != n_classes:
         logger.warning("Mismatch between n_classes and length of class_names. Using default names.")
         class_names = [f"Class {i}" for i in range(n_classes)]


    # Plot ROC curve for each class
    for class_id in range(n_classes):
         display_name = class_names[class_id]
         RocCurveDisplay.from_predictions(
             targets == class_id,  # True labels for this class vs rest
             predictions[:, class_id], # Predicted probabilities for this class
             name=f"{display_name}", # Use class name
             ax=ax,
             # alpha=0.8 # Optional: make lines slightly transparent
         )


    ax.plot([0, 1], [0, 1], 'k--', label='Chance level (AUC = 0.5)') # Add chance line
    ax.set_title(f'One-vs-Rest ROC Curves - {model_name}')
    ax.set_xlabel('False Positive Rate')
    ax.set_ylabel('True Positive Rate')
    ax.legend()
    ax.grid(True)
