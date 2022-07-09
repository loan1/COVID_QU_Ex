from torch.utils.data import Dataset, DataLoader
from PIL import Image
import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os

class ImageDataset(Dataset):
    def __init__(self,csv, train_path, transform = None): # 'Initialization'
        self.csv=csv 
        self.transform=transform    
        self.image_names=self.csv[:]['file_name']# [:] lấy hết số cột số hàng của bảng
        self.labels= self.csv[:]['label']
        self.train_path = train_path 
  
    def __len__(self):  # 'Denotes the total number of samples'
        return len(self.image_names)

    def __getitem__(self,index): # 'Generates one sample of data'
        ''' repr: xu li duong dan co dau cach
            [1:-1]: bo ' ' dau va cuoi duong dan '''
        img_path = repr(self.train_path + '/' + self.labels[index] + '/images/' + self.image_names.iloc[index])[1:-1]
        mask_path = repr(self.train_path + '/' + self.labels[index] + '/lung masks/' + self.image_names.iloc[index])[1:-1]  

        image=Image.open(img_path).convert('L') # https://viblo.asia/p/series-pandas-dataframe-phan-tich-du-lieu-cung-pandas-phan-3-WAyK8AMEZxX
        mask = Image.open(mask_path).convert('L')

        image = np.array(image, dtype=np.float32)
        mask = np.array(mask, dtype=np.float32)

        mask[mask==255] = 1 
        image = cv2.GaussianBlur(image, (3,3), cv2.BORDER_DEFAULT)
        # print('1: ',image.shape) #(256, 256)
        # print('2: ', np.expand_dims(image,0).shape) #(1, 256, 256)
        # mo rong them 1 dim 0 sau do chuyen dim 0 ra sau
        image = np.expand_dims(image,0).transpose(1,2,0)#### (256, 256, 1) 
        # print('3: ', image.shape)
        if self.transform != None:
            aug = self.transform(image = image, mask = mask)
            image=aug['image']
            mask = aug['mask']    
     
        label = self.labels[index]
        file_name = self.image_names[index]

        return image, mask, label, file_name 
        
def show_img(img, mask,fn):
    plt.subplot(1,2,1)
    plt.imshow(img, cmap='gray')
    plt.title(fn)

    plt.subplot(1,2,2)
    plt.imshow(mask, cmap='gray')
    plt.title(fn)
    plt.show()



class DatasetPredict(Dataset):
    def __init__(self, img_folder,  transform = None): # 'Initialization'

        self.img_folder = img_folder
        self.transform = transform
  
    def __len__(self):  # 'Denotes the total number of samples'
        return len(os.listdir(self.img_folder))

    def __getitem__(self,index): # 'Generates one sample of data'      

        images_list = os.listdir(self.img_folder)
        images_name = images_list[index]

        images = cv2.imread(self.img_folder +  images_name, 0) # grey 

        images = np.array(images, dtype=np.float32) # đoi qua numpy array kiểu float 32

        image = cv2.GaussianBlur(image, (3,3), cv2.BORDER_DEFAULT)    

        if self.transform != None:
            aug = self.transform(image = images)            
            images = aug['image']

        return images, images_name # chua 1 anh + tên

if __name__ == '__main__':
    train_csv = pd.read_csv('/media/trucloan/Data/Research/Andy_Le/lung_segmentCOVIDx/COVID_QU_Ex/Lung Segmentation Data/Lung Segmentation Data/train.csv')
    train_dataset = ImageDataset(train_csv, '/media/trucloan/Data/Research/Andy_Le/lung_segmentCOVIDx/COVID_QU_Ex/Lung Segmentation Data/Lung Segmentation Data/Train')
    train_dataloader = DataLoader(train_dataset, batch_size=32, shuffle=False)

    for idx in range(12):
        img, msk, lb, fn = next(iter(train_dataloader))
        # show_img(img,msk,fn)
        print(img)

    # pa = '/media/trucloan/Data/Research/Andy_Le/lung_segmentCOVIDx/COVID_QU_Ex/Lung Segmentation Data/Lung Segmentation Data/Train/COVID-19/images/covid_1.png'
    # pa = repr(pa)
    # print(pa)
    # Image.open(r'/media/trucloan/Data/Research/Andy_Le/lung_segmentCOVIDx/COVID_QU_Ex/Lung Segmentation Data/Lung Segmentation Data/Train/COVID-19/images/covid_1.png')



    # transfm = A.Compose([
    #     A.Resize(512,512),
    #     A.Normalize(mean = [0.5],  std = [0.5]),
    #     ToTensorV2()
    # ])


# def get_mean_std(loader):
#     channels_sum, channels_sqrd_sum, num_batches = 0, 0, 0

#     for data, _ in tqdm(loader):
#         channels_sum += torch.mean(data, dim=[0, 2, 3])
#         channels_sqrd_sum += torch.mean(data ** 2, dim=[0, 2, 3])
#         num_batches += 1

#     mean = channels_sum / num_batches
#     std = (channels_sqrd_sum / num_batches - mean ** 2) ** 0.5

#     return mean, std


# mean, std = get_mean_std(train_loader)
# print(mean)
# print(std)