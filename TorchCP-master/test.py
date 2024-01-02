from torchcp.classification.scores import THR,APS,SAPS,RAPS
from torchcp.classification.predictors import SplitPredictor,ClusterPredictor,ClassWisePredictor
import torchvision.datasets as datasets
import torch
from torch.utils.data import DataLoader
from torchvision import transforms
import torchvision
import torch.nn as nn


import random
import numpy as np

from torchvision.datasets import Flowers102

random.seed(2)
np.random.seed(2)
torch.manual_seed(2)
torch.cuda.manual_seed(2)

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")



model = torchvision.models.vit_b_16(pretrained=True)


for param in model.parameters():
    param.requires_grad = False


num_features = model.heads[0].in_features


model.heads[0] = nn.Linear(num_features, 102)


normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    normalize,
])


cal_dataset = Flowers102(root='./data', split='train', download=True, transform=transform)
test_dataset = Flowers102(root='./data', split='val', download=True, transform=transform)

cal_dataloader = DataLoader(cal_dataset, batch_size=32, shuffle=True, num_workers=2)
test_dataloader = DataLoader(test_dataset, batch_size=32, shuffle=False, num_workers=2)

print('dataloader prepared')


model.eval()

score_funcs = [THR(), APS(),SAPS(weight=0.2),RAPS(penalty=1)]  #SAPS, RAPS() not available
predictor_classes = [SplitPredictor, ClusterPredictor, ClassWisePredictor]  


alpha = 0.1

for score_func in score_funcs:
    # define a score function. Optional: THR, APS, SAPS, RAPS
    score_function = score_func
    for predictor_class in predictor_classes:
        # define a conformal prediction algorithm. Optional: SplitPredictor, ClusterPredictor, 
        # ClassWisePredictor
        predictor = predictor_class(score_function , model)  
        # calibration process
        predictor.calibrate(cal_dataloader, alpha)
        print('finish calibrate')


        result = predictor.evaluate(test_dataloader)
        print(result)

        '''save result as txt'''
        with open('result_fashionMNIST.txt', 'a') as f:
            f.write('score func: '+str(score_function)+'\n')
            f.write('alpha: '+str(alpha)+'\n')
            f.write('predictor: '+str(predictor)+'\n')
            f.write(str(result)+'\n')  
            f.write('\n')









