 
# https://github.com/qubvel/segmentation_models.pytorch
# https://github.com/IlliaOvcharenko/lung-segmentation
# https://www.kaggle.com/pezhmansamadi/lung-segmentation-torch

import torch
import time
from operator import add
import sys


def train_loop(model, loader, optimizer, scheduler, loss_fn, metric_fn, device):

    epoch_loss = 0.0
    metrics_score = [0.0, 0.0, 0.0, 0.0, 0.0]
    steps = len(loader)
    
    model.train()

    for i, (x, y, _, _) in enumerate (loader):
        x = x.to(device)
        y = y.float().unsqueeze(1).to(device)

        optimizer.zero_grad()

        y_pred = model(x) # gpu cuda

        loss = loss_fn(y_pred, y) # comboloss: BCE dice focal
        loss.backward() # 
        
        score = metric_fn(y_pred, y)
        metrics_score = list(map(add, metrics_score, score))
        
        optimizer.step()
        learning_rate = optimizer.param_groups[0]['lr']     

        epoch_loss += loss.item()
        
        sys.stdout.flush()
        sys.stdout.write('\r Step: [%2d/%2d], loss: %.4f - acc: %.4f' % (i, steps, loss.item(), score[1]))
    scheduler.step()
    
    sys.stdout.write('\r')

    epoch_loss = epoch_loss/len(loader)
    
    epoch_jaccard = metrics_score[0]/len(loader)
    epoch_acc = metrics_score[1]/len(loader)
    epoch_dice = metrics_score[2] / len(loader)
    epoch_recall = metrics_score[3] / len(loader)
    epoch_precision = metrics_score[4] / len(loader)
    
    return epoch_loss, epoch_jaccard, epoch_dice, epoch_recall, epoch_precision, epoch_acc, learning_rate,  

def evaluate(model, loader, loss_fn, metric_fn, device):
    epoch_loss = 0.0
    metrics_score = [0.0, 0.0, 0.0, 0.0, 0.0]

    model.eval()
    with torch.no_grad():
        for x, y, _, _ in loader:
            x = x.to(device)

            y = y.float().unsqueeze(1).to(device)

            y_pred = model(x)
            loss = loss_fn(y_pred, y)
            
            score = metric_fn(y_pred, y)
            metrics_score = list(map(add, metrics_score, score))
            
            epoch_loss += loss.item()

        epoch_loss = epoch_loss / len(loader)
        
        epoch_jaccard = metrics_score[0] / len(loader)
        epoch_acc = metrics_score[1] / len(loader)
        epoch_f1 = metrics_score[2] / len(loader)
        epoch_recall = metrics_score[3] / len(loader)
        epoch_precision = metrics_score[4] / len(loader)
    
    return epoch_loss, epoch_jaccard, epoch_f1, epoch_acc, epoch_recall, epoch_precision

def epoch_time(start_time, end_time):
    elapsed_time = end_time - start_time
    elapsed_mins = int(elapsed_time / 60)
    elapsed_secs = int(elapsed_time - (elapsed_mins * 60))
    return elapsed_mins, elapsed_secs

def fit (model, train_dl, valid_dl, optimizer, scheduler, start_epochs, end_epochs, loss_fn, metric_fn, checkpoint_path, model_path, device, train_list =[], val_list=[]):
    """ fiting model to dataloaders, saving best weights and showing results """

    learning_rate =[]

    best_val_loss = float("inf")
    patience = 8 

    since = time.time()
    for epoch in range (start_epochs, end_epochs):
        ts = time.time()
        
        train_loss, jaccard, dice,recall,precision, acc,  lr = train_loop(model, train_dl, optimizer, scheduler, loss_fn, metric_fn, device)
        val_loss, val_jaccard, f1, val_acc, val_recall, val_precision = evaluate(model, valid_dl, loss_fn, metric_fn, device)        

        train_list.append({'epoch':epoch,'train_loss': train_loss,'jaccard':jaccard, 'dice': dice, 'recall': recall, 'precision': precision, 'accuracy': acc})
        val_list.append({'epoch':epoch,'val_loss': val_loss,'val_jaccard':val_jaccard, 'val_dice': f1, 'recall': val_recall, 'precision': val_precision, 'accuracy': val_acc})
        learning_rate.append(lr)
        
        te = time.time() 

        epoch_mins, epoch_secs = epoch_time(ts, te)
        
        # print ('Epoch [{}/{}], loss: {:.4f} - jaccard: {:.4f} - acc: {:.4f}  - val_loss: {:.4f} - val_jaccard: {:.4f} - val_acc: {:.4f}'.format (epoch + 1, epochs, loss, jaccard, acc, val_loss, val_jaccard, val_acc))
        print ('Epoch [{}/{}], loss: {:.4f} - jaccard: {:.4f} - acc: {:.4f}  - f1: {:.4f} - recall: {:.4f} - precision: {:.4f}'.format (epoch + 1, end_epochs, train_loss, jaccard, acc, f1, recall, precision))
        print ('val_loss: {:.4f} - val_jaccard: {:.4f} - val_acc: {:.4f} - val_f1: {:.4f} - val_recall: {:.4f} - val_precision: {:.4f}'.format (val_loss, val_jaccard, val_acc, f1, val_recall, val_precision))
        print(f'Time: {epoch_mins}m {epoch_secs}s')
    
        period = time.time() - since
        print('Training complete in {:.0f}m {:.0f}s'.format(period // 60, period % 60))

        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'train_list': train_list, 
            'val_list': val_list,
            }, checkpoint_path)


        if val_loss < best_val_loss:
            count = 0
            data_str = f"===> Valid loss improved from {best_val_loss:2.4f} to {val_loss:2.4f}. Saving checkpoint: {checkpoint_path}"
            print(data_str)
            best_val_loss = val_loss
            # save_checkpoint(model.state_dict(), checkpoint_path)
            torch.save(model.state_dict(), model_path) #save checkpoint           
            
        else:
            count += 1
            if count >= patience:
                print('Early stopping!')
                return train_list, val_list
    
    
    return train_list, val_list
