import torch.nn as nn
from torchvision import models
import torch
import os
from whole_classifier_model import Bottleneck,Resnet50

def initialize_data_loader(batch_size,workers,root,aug = False) -> Tuple[DataLoader, DataLoader, DataLoader]:

    
    train = os.path.join(root,"data/train.csv")
    validation =os.path.join(root,"data/validation.csv")
    test = os.path.join(root,"data/test.csv")
   

    normalize = T.Normalize(mean=[0.2030],
                                     std=[0.2646])

    augmentation = T.Compose([normalize,T.RandomHorizontalFlip(), T.RandomVerticalFlip(),T.RandomRotation(degrees=25),
                        T.RandomAffine(degrees=0, scale=(0.8, 0.99)),T.RandomResizedCrop(size=(1152,896),scale=(0.8,0.99)),
                        MyIntensityShift(shift= [80,120]), T.RandomAffine(degrees=0, shear=12)])


    if aug:
        train_dataset= CBIS_MAMMOGRAM(train, transform = augmentation)
    
    else: 
        train_dataset = CBIS_MAMMOGRAM(train, transform = T.Compose([T.ToTensor(),normalize]))
    #Normalizing the validation set
    validation_dataset = CBIS_MAMMOGRAM(validation, transform = T.Compose([T.ToTensor(),normalize]))
    #Normalizing the test set
    test_dataset = CBIS_MAMMOGRAM(test, transform = T.Compose([T.ToTensor(),normalize]))

  
    train_loader = DataLoader(train_dataset, batch_size= batch_size, sampler = sampler,num_workers = 8 )
    val_loader = DataLoader(validation_dataset, batch_size = batch_size)
    test_loader = DataLoader(test_dataset, batch_size = batch_size) 



    return train_loader,val_loader,test_loader

def Initialize_model(device_id,root):
    model = Resnet50(Bottleneck, layers=[2,2], use_pretrained=True, root=root)
    model.cuda(device_id)
    cudnn.benchmark = True
    model = DistributedDataParallel(model, device_ids=[device_id],find_unused_parameters=False)

    criterion = nn.CrossEntropyLoss().cuda(device_id)

    return model, criterion
