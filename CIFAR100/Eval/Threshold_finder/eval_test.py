import os
import argparse
import torch
import torch.nn as nn
import torchvision.datasets as datasets
import torch.utils.data as data
import torchvision.transforms as transforms
import numpy as np
import sys
import models
from models.resnet import VAE
from torch.utils.data.sampler import SubsetRandomSampler
import torchvision
import torch.optim as optim
from torchvision import datasets, transforms

parser = argparse.ArgumentParser()
parser.add_argument('--data_dir', type=str, default='../../data')
parser.add_argument('--model', type=str, default='./model_test.pt')
parser.add_argument('--n_ex', type=int, default=10000)
parser.add_argument('--save_dir', type=str, default='./results_threshold_finder')
parser.add_argument('--batch_size', type=int, default=1000)
parser.add_argument('--log_path', type=str, default='./log_file.txt')
parser.add_argument('--count_FP', type=int, default=1001)
parser.add_argument('--total_SS_steps', type=int, default=100)
parser.add_argument('--sample_std', type=int, default=2)
parser.add_argument('--no_state_dict', type=bool, default=True)


args = parser.parse_args()

def threshold_finder(args,model,x_val, y_val):
    x_val_true  = x_val
    counter_arr = np.zeros(args.total_SS_steps)
    y_val_true = y_val
    for k in range(int(len(x_val_true)/args.batch_size)):
        x_val = x_val_true[k*args.batch_size:(k+1)*args.batch_size].cuda()
        y_val = y_val_true[k*args.batch_size:(k+1)*args.batch_size].cuda()
        maxa=np.zeros(len(x_val))
        output = np.zeros(len(x_val))
        lst_ind=np.zeros([len(x_val),100])	
        for i in range(args.total_SS_steps):
            answer = model(x_val)
            ans=answer[0].max(dim=1)[1]
            for j in range(len(ans)):
                lst_ind[j][ans[j]]+=1
                if lst_ind[j][ans[j]]>maxa[j]:
                    maxa[j] = lst_ind[j][ans[j]]
                    output[j] = ans[j]
        y_val = y_val.detach().cpu().numpy()
        for i in range(args.total_SS_steps):
            ls=[]
            counter_rejected=0
            break_counter=counter_arr[i]
            counter_accepted=0
            for j in range(len(maxa)):            
                if maxa[j]>i:   
                    if y_val[j]==output[j]:
                        counter_accepted+=1
                else:
                    counter_rejected+=1
                    if y_val[j]==output[j]:
                        break_counter+=1
            counter_arr[i]=break_counter
            if break_counter>=args.count_FP and k==int(len(x_val_true)/args.batch_size)-1:
                print("The real closest count_FP is",break_counter)
                break
    return i-1



if __name__ == '__main__':
    # load model
    sys.path.append("./")
    model  = VAE(args.sample_std).cuda()
    model = torch.nn.DataParallel(model)
    if not args.no_state_dict:
       model.load_state_dict(torch.load(args.model)["state_dict"])
    else:
       model.load_state_dict(torch.load(args.model))
    model.eval()
    model.cuda()

    #load data
    TRAIN_BATCH_SIZE = args.batch_size
    VAL_BATCH_SIZE   = args.batch_size 
    TEST_BATCH_SIZE   = args.batch_size
    transform_train = transforms.Compose([
        transforms.RandomCrop(size=32,padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),])

    transform_test = transforms.Compose([
        transforms.ToTensor(),])

    train_set  = torchvision.datasets.CIFAR100(root=args.data_dir, train=True , download=True, transform=transform_train)
    val_set    = torchvision.datasets.CIFAR100(root=args.data_dir, train=True , download=True, transform=transform_test)
    test_set   = torchvision.datasets.CIFAR100(root=args.data_dir, train=False, download=True, transform=transform_test)

    # Split training into train and validation
    train_size = 47500
    valid_size = 2500
    test_size  = 10000
    train_indices = list(range(50000))
    val_indices = []
    count = np.zeros(100)
    for index in range(len(train_set)):
        _, target = train_set[index]
        if(np.all(count==25)):
            break
        if(count[target]<25):
            count[target] += 1
            val_indices.append(index)
            train_indices.remove(index)
        
    print("Overlap indices:",list(set(train_indices) & set(val_indices)))
    print("Size of train set:",len(train_indices))
    print("Size of val set:",len(val_indices))
    test_loader   = torch.utils.data.DataLoader(test_set,batch_size=TEST_BATCH_SIZE)
    print('CIFAR100 dataloader: Done')  

    #####  DATALOADER DONE #####

    # create save dir
    if not os.path.exists(args.save_dir):
        os.makedirs(args.save_dir)
    
    l = [x for (x, y) in test_loader]
    x_val = torch.cat(l, 0)
    l = [y for (x, y) in test_loader]
    y_val = torch.cat(l, 0)

    with torch.no_grad():
        thresh=0
        print("Running the threshold_finder rejection ")
        threshold_value = threshold_finder(args,model,x_val[:args.n_ex], y_val[:args.n_ex])
        print("The threshold value is:",threshold_value)
        thresh=thresh+threshold_value
        THRESHOLD = '../threshold.txt'
        log_file = open(THRESHOLD,'w')
        log_file.write(str(thresh))

