import albumentations  as A
from albumentations.pytorch.transforms import ToTensorV2
import numpy as np

augs = A.Compose([
        A.Resize(256, 256), 
        A.HorizontalFlip(p=0.5),
        A.OneOf([
            A.ElasticTransform(alpha=120, sigma=120 * 0.05, alpha_affine=120 * 0.03),  # phép biến đổi co giãn
            A.Rotate(limit=15),
        ], p = 0.1),
  
        A.Normalize(mean = [0.5128], std =[0.2220]),
        ToTensorV2()   
    ])

transfms = A.Compose([
    A.Resize(256,256),
    A.Normalize(mean = [0.5128], std =[0.2220]), 
    ToTensorV2() 
])

def img_de_normalize (img, mask):
        
    img = np.squeeze(img)
    img = img*0.2220 + 0.5128
    mask = mask*0.220+0.5128
    img= np.clip(img, 0, 1)
    mask = np.clip(mask, 0,1)
    
    return img, mask

# dataset





