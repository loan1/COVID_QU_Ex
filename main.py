# import modules
from script.Train_data import augs, transfms
from script.train import fit
from script.visualize import plot_IoU_DSC, plot_acc_loss, plot_loss
from script.utils import load_checkpoint, seed_everything, ComboLoss, calculate_metrics, set_dataloader, set_model, set_model
from script.test import test
from script.predict import dataloaderPre, predict, postprocess
import script.models as models

# import lib
import argparse
import torch
import numpy as np
import os
from PIL import Image
import matplotlib.pyplot as plt
from tqdm import tqdm


def get_opt():
    parser = argparse.ArgumentParser()
    parser.add_argument('--batch_size', default=8, type=int)
    parser.add_argument('--start_epoch', default=0, type=int)
    parser.add_argument('--end_epoch', default= 40, type=int)
    parser.add_argument('--lr', default=1e-4, type=float)
    parser.add_argument('--checkpoint_path', default='./model/checkpoint.pt', type=str)
    parser.add_argument('--model_path', default='./model/UNet.pt', type=str)
    parser.add_argument('--first_train', default=1, type=int)
    opt = parser.parse_args()
    return opt

def visualize_train(train_list, val_list):

    train_loss, val_loss, train_acc, val_acc = [], [], [], []
    train_IoU, val_IoU, train_dice, val_dice = [], [], [], []

    for idx in range(len(train_list)):
        train_loss.append(train_list[idx]['train_loss'])
        val_loss.append(val_list[idx]['val_loss'])
        train_acc.append(train_list[idx]['accuracy'])
        val_acc.append(val_list[idx]['accuracy'])
        train_IoU.append(train_list[idx]['jaccard'])
        val_IoU.append(val_list[idx]['val_jaccard'])
        train_dice.append(train_list[idx]['dice'])
        val_dice.append(val_list[idx]['val_dice'])
    # print(val_loss)   

    # VE LOSS_ACC
    plot_acc_loss(train_loss, val_loss, train_acc, val_acc, './visualize/testloss_acc.png')  
    plot_IoU_DSC(train_IoU, val_IoU,train_dice, val_dice, './visualize/IoU_DSC')


def train( device, model, optimizer, scheduler, augs, transfms, start_epochs, end_epochs, loss_fn, batch_size, checkpoint_path, model_path, first = True, train_list =[], val_list = []):
    if first == 1:    
        train_dataloader, val_dataloader, _ = set_dataloader(batch_size, augs, transfms)
    else:
        model, optimizer, start_epochs, train_list, val_list = load_checkpoint(checkpoint_path, model, optimizer)
    train_list, val_list = fit(model, train_dataloader, val_dataloader, optimizer, scheduler, start_epochs, end_epochs, loss_fn, calculate_metrics, checkpoint_path, model_path, device, train_list, val_list)
    
    # plot result
    visualize_train(train_list, val_list)

    return train_list, val_list


def save_filename(data, path):
        #################################
    # TAO FILE TXT LUU TEN ANH #####
    ################################
    list_name =[]
    # for _, name in tqdm(dataloaderPre(opt.batch_size, transfms)['Positive']):
    for _, name in tqdm(data):
        list_name.append(name)

    print(len(list_name)) # 438
    print(len(list_name[437])) # 8

    res = []
    for i in tqdm(range(len(list_name)-1)):
        for idx in range(opt.batch_size):
            # print(i, idx, end= ' ')
            res.append(list_name[i][idx])
    
    for i in range(8): #xu li phan le trong batch cuoi
        res.append(list_name[len(list_name)-1][i])

    print(len(res))

    # np.savetxt('./visualize/FPN_DenseNet121/lung_mask/Positive/filenamePos.txt', res, fmt = '%s')
    np.savetxt(path, res, fmt = '%s')
    
# save lung mask
def save_lungmask (file_name, path, path_np):
    # images_np = np.load('./visualize/FPN_DenseNet121/images.npy', allow_pickle=True)
    y = np.load(path_np, allow_pickle=True)
    
    # print(type(y)) # <class 'numpy.ndarray'>
    # print(y.shape) # (2,)

    # list_name = np.loadtxt('./visualize/FPN_DenseNet121/lung_mask/Positive/filenamePos.txt', dtype = list)
    list_name = np.loadtxt(file_name, dtype = list)
    # print(len(dataloaderPre(opt.batch_size, transfms)['Positive'])) #500
    le = len(dataloaderPre(opt.batch_size, transfms)['Positive']) # 
    # print(list_name[15993]) # A358591-02-02-1901-NA-CHEST_AP_PORT-74666-3.000000-AP-63366-1-1.jpg
    # print(y[499][25])

    for i in tqdm(range(le)):   # 
        for idx in range(opt.batch_size):    
            ret = postprocess(y[i][idx])       
            # plt.imsave('./visualize/FPN_DenseNet121/lung_mask/Positive/' + list_name[i*opt.batch_size+idx], y[i][idx])
            # plt.imsave(path + list_name[i*opt.batch_size+idx], y[i][idx])
            plt.imsave(path + list_name[i*opt.batch_size+idx], ret)

    plt.close('all')

if __name__ == '__main__':
    seed = 262022
    seed_everything(seed)

    opt = get_opt()
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    model = models.list_model_FPN[5].to(device)

    optimizer, scheduler, loss_fn = set_model(device, opt.lr, model)

    # # TRAIN
    # train_list, val_list = train(device, model, optimizer, scheduler, augs, transfms, opt.start_epoch, opt.end_epoch, loss_fn, opt.batch_size, opt.checkpoint_path, opt.model_path, opt.first_train)

    # LOAD CHECKPOINT
    model, optimizer, start_epochs, train_list, val_list = load_checkpoint(opt.checkpoint_path, model, optimizer)
    
    
    # # TEST 
    # model.load_state_dict(torch.load(opt.model_path))
    # _,_, test_dataloader = set_dataloader(opt.batch_size, augs, transfms)

    # image, y_true, y_pred, test_dict = test(test_dataloader, device, model, calculate_metrics) 
    # print(test_dict)
    
    # ghi ket qua
    # with open('./visualize/test_result.txt', 'w') as wf:
    #     wf.writelines(str(test_dict.items()))

    # PREDICT POSITIVE
    # model.load_state_dict(torch.load(opt.model_path)) #load model
    # images, y_predict = predict(dataloaderPre(opt.batch_size, transfms)['Positive'], model, device) #predict

    # # luu ket qua inference
    # np.save('./visualize/FPN_DenseNet121/images.npy',images)
    # np.save('./visualize/FPN_DenseNet121/y_predict.npy',y_predict)

    # # save file name
    # save_filename(dataloaderPre(opt.batch_size, transfms)['Positive'], './visualize/FPN_DenseNet121/lung_mask/Positive/filenamePos.txt')

    # #save lung mask
    save_lungmask (file_name ='./visualize/FPN_DenseNet121/lung_mask/Positive/filenamePos.txt', path = './visualize/FPN_DenseNet121/lung_mask/PositivePos/', path_np='./visualize/FPN_DenseNet121/y_predict.npy')

    # PREDICT NEGATIVE
    # model.load_state_dict(torch.load(opt.model_path)) #load model
    # images, y_predict = predict(dataloaderPre(opt.batch_size, transfms)['Negative'], model, device) #predict

    # luu ket qua inference
    # np.save('./visualize/FPN_DenseNet121/imagesN.npy',images)
    # np.save('./visualize/FPN_DenseNet121/y_predictN.npy',y_predict)

    # save file name
    # save_filename(dataloaderPre(opt.batch_size, transfms)['Negative'], './visualize/FPN_DenseNet121/lung_mask/Negative/filenameNeg.txt')

    #save lung mask
    # save_lungmask (file_name ='./visualize/FPN_DenseNet121/lung_mask/Negative/filenameNeg.txt', path = './visualize/FPN_DenseNet121/lung_mask/NegativePos/', path_np='./visualize/FPN_DenseNet121/y_predictN.npy')
    







    


    

