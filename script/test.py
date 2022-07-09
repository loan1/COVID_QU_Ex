# import lib

import torch
import numpy as np
from operator import add
from tqdm import tqdm

def test(dataloader, device, model, metric_fn):   
    with torch.no_grad():
        image, y_true, y_predict = [], [], []
        metrics_score = [0.0, 0.0, 0.0, 0.0, 0.0]
        test_dict = {}
        model.eval()
        for x, y, _, _ in tqdm(dataloader):
            x = x.to(device)
            y = y.float().unsqueeze(1).to(device)           

            #######################
            #### METRIC###########
            #######################
            y_pred = model(x)
            score = metric_fn(y_pred, y)
            metrics_score = list(map(add, metrics_score, score))  

            ############################################################################

            pred = y_pred.cpu().numpy() # mask output 
            ynum = y.cpu().numpy()  # mask label

            pred = pred.reshape(len(pred), 256, 256) # 4, 256, 256
            ynum = ynum.reshape(len(ynum), 256, 256)

            pred = pred > 0.1 #threshold
            pred = np.array(pred, dtype=np.uint8)

            y_true.append(ynum)    
            y_predict.append(pred)

            # chuyển đổi ngược lại với transform
            x = x.cpu().numpy()
            x = x.reshape(len(x), 256, 256)
            x = x*0.3 + 0.59
            x = np.squeeze(x)
            x = np.clip(x, 0, 1)

            image.append(x)

        epoch_jaccard = metrics_score[0]/len(dataloader)
        epoch_acc = metrics_score[1]/len(dataloader)
        epoch_f1 = metrics_score[2]/len(dataloader)
        epoch_recall = metrics_score[3]/len(dataloader)
        epoch_precision = metrics_score[4]/len(dataloader)   

        test_dict = {'jaccard': epoch_jaccard, 'acc': epoch_acc, 'f1': epoch_f1, 'recall': epoch_recall, 'precision': epoch_precision}# {}    
        
        print ('jaccard: {:.4f} - acc: {:.4f} - val_f1: {:.4f} - recall: {:.4f} - precision: {:.4f}'.format (epoch_jaccard, epoch_acc, epoch_f1, epoch_recall, epoch_precision))

    return image, y_true, y_predict, test_dict

#########################################################################################

def save_np(path, image, y_true, y_prect ):
    np.save(path + '/images.npy',image)
    np.save(path + '/masks.npy',y_true)
    np.save(path + '/predict.npy',y_prect)

    
############################################################################################################        

def load_np(path):
    images_np = np.load(path + '/images.npy', allow_pickle = True)
    masks_np = np.load(path + '/masks.npy', allow_pickle = True)
    y_prect = np.load(path +  '/predict.npy', allow_pickle = True)    
    return images_np, masks_np, y_prect

#############################################################################################################