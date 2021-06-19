'''
Code for training of robust networks with Feature Level Stochastic Smoothing
The code in this file is based on the publically available TRADES and AWP GitHub Repo 
'''


from __future__ import print_function
import os
import argparse
import torch
import time
import torch.nn as nn
import torch.nn.functional as F
import torchvision
from torch.autograd import Variable
import torch.optim as optim
from torchvision import datasets, transforms
from arch.RN_FC import *
import numpy as np
from torch.autograd.gradcheck import zero_gradients
from torch.utils.data.sampler import SubsetRandomSampler
import sys
from utils import Bar, Logger, AverageMeter, accuracy
from utils_awp import TradesAWP

parser = argparse.ArgumentParser(description='Adversarial Training with Feature Level Stochastic Smoothing')
parser.add_argument('--batch-size', type=int, default=128, metavar='N',
                    help='input batch size for training (default: 64)')
parser.add_argument('--test-batch-size', type=int, default=512, metavar='N',
                    help='input batch size for testing (default: 512)')
parser.add_argument('--epochs', type=int, default=120, metavar='N',
                    help='number of epochs to train')
parser.add_argument('--weight-decay', '--wd', default=5e-4,
                    type=float, metavar='W')
parser.add_argument('--lr', type=float, default=0.1, metavar='LR',
                    help='learning rate')
parser.add_argument('--data_dir', type=str, default='../data')
parser.add_argument('--momentum', type=float, default=0.0, metavar='M',
                    help='SGD momentum')
parser.add_argument('--no-cuda', action='store_true', default=False,
                    help='disables CUDA training')
parser.add_argument('--epsilon', default=8/255.0,
                    help='perturbation')

parser.add_argument('--num-steps', default=10,
                    help='perturb number of steps')
parser.add_argument('--step-size', default=2/255.0,
                    help='perturb step size')
parser.add_argument('--seed', type=int, default=1, metavar='S',
                    help='random seed (default: 1)')
parser.add_argument('--log-interval', type=int, default=100, metavar='N',
                    help='how many batches to wait before logging training status')
parser.add_argument('--model-dir', default='./model-cifar-ResNet',
                    help='directory of model for saving checkpoint')
parser.add_argument('--save-freq', '-s', default=1, type=int, metavar='N',
                    help='save frequency')

parser.add_argument('--awp-gamma', default=0.005, type=float,
                    help='whether or not to add parametric noise')
parser.add_argument('--awp-warmup', default=10, type=int,
                    help='We could apply AWP after some epochs for accelerating.')

args = parser.parse_args()

# settings
model_dir = args.model_dir
if not os.path.exists(model_dir):
    os.makedirs(model_dir)
use_cuda = not args.no_cuda and torch.cuda.is_available() 
torch.manual_seed(args.seed)
device = torch.device("cuda" if use_cuda else "cpu")
kwargs = {'num_workers': 1, 'pin_memory': True} if use_cuda else {}


TRAIN_BATCH_SIZE = args.batch_size
VAL_BATCH_SIZE   = args.test_batch_size
TEST_BATCH_SIZE   = args.test_batch_size
transform_train = transforms.Compose([
        transforms.RandomCrop(size=32,padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),])

transform_test = transforms.Compose([
        transforms.ToTensor(),])

train_set  = torchvision.datasets.CIFAR100(root=args.data_dir, train=True , download=True, transform=transform_train)
val_set    = torchvision.datasets.CIFAR100(root=args.data_dir, train=True , download=True, transform=transform_test)
test_set   = torchvision.datasets.CIFAR100(root=args.data_dir, train=False, download=True, transform=transform_test)

# Split training into train and balanced class 2500 images validation
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
#get data loader ofr train val and test
train_loader = torch.utils.data.DataLoader(train_set,batch_size=TRAIN_BATCH_SIZE ,sampler=SubsetRandomSampler(train_indices))
val_loader   = torch.utils.data.DataLoader(val_set,sampler = SubsetRandomSampler(val_indices),batch_size=VAL_BATCH_SIZE)
test_loader   = torch.utils.data.DataLoader(test_set,batch_size=TEST_BATCH_SIZE)
print('CIFAR100 dataloader: Done')  
if not os.path.exists("./results_train"):
    os.makedirs("./results_train")
LOSS_TRAIN = "./results_train" + '/Train_loss.txt'
ACC_PGD10_TRAIN = "./results_train" + '/Train_PGD_10.txt'
ACC_CLEAN_TRAIN = "./results_train" + '/Train_CLEAN.txt'
loss = nn.CrossEntropyLoss()
####################################
#Training:

#Attack function
def pgd_loss(model,
                x_natural,
                y,
                optimizer,
                step_size=2/255,
                epsilon=8/255,
                perturb_steps=10,
                distance='l_inf',epc=100):
    #Making the model into eval mode as the attack begins
    model.eval()
    batch_size = len(x_natural)



    # generation of the first adversarial example begins 
    
    noise  = torch.FloatTensor(np.random.uniform(-epsilon,epsilon,x_natural.size())).cuda()
    x_adv = (x_natural.detach() + noise.detach()).clamp(0,1)

    if distance == 'l_inf':
        for index in range(perturb_steps):
            x_adv.requires_grad_()
            with torch.enable_grad():

                #Basically sampling for the very first step of the attack and then using same sampled noise for remainig steps. 
                if index==0:
                    #Getting the logits,mean vector, variace vectors and noise vectors.
                    adv_logits,mu_adv,logvar_adv,eps_adv = model(x_adv)
                    loss_ce = nn.CrossEntropyLoss()(adv_logits, y)
                    
        
                #Now using the same sampled noise in the first step; for remaining steps.
                #no_sample=True means it will not sample epsilon in latent space but rather take is as an input which is epso.
                else: 
                    adv_logits,mu_adv,logvar_adv,eps_adv = model(x_adv,no_sample=True,epso=eps_adv)
                    loss_ce = nn.CrossEntropyLoss()(adv_logits, y)
                    
            
            #Generating x_adv
            loss = loss_ce 
            grad = torch.autograd.grad(loss, [x_adv])[0]
            x_adv = x_adv.detach() + step_size * torch.sign(grad.detach())
            x_adv = torch.min(torch.max(x_adv, x_natural - epsilon), x_natural + epsilon)
            x_adv = torch.clamp(x_adv, 0.0, 1.0)

    #First adversarial example generated

    #Inner maximization ends
    #Shifting the model back to the train mode
    model.train()
    model.module.fc1_encoder.eval() 

    #Outer minimization begins
    x_adv = Variable(torch.clamp(x_adv, 0.0, 1.0), requires_grad=False)
    # zero gradient
    return x_adv,eps_adv

#Train function
def train(args, model, device, train_loader, optimizer, epoch,scheduler,awp_adversary):
    model.train()
    model.module.fc1_encoder.eval() 
    tot_loss=0

    epoch_accuracy_pgd = torch.zeros(1).cuda()
    epoch_accuracy_clean = torch.zeros(1).cuda()
    start_time = time.time()
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        batch_size = len(data)
        optimizer.zero_grad()

        # calculate robust loss
        #The first step of alternate iterations that is adversarial training step begins

        epsilon = 8/255.0
        x_adv,eps_adv = pgd_loss(model=model,
                           x_natural=data,
                           y=target,
                           optimizer=optimizer,
                           step_size=args.step_size,
                           epsilon=epsilon,
                           perturb_steps=args.num_steps,
                           epc=epoch)
        optimizer.zero_grad()
        if epoch >= args.awp_warmup:
            awp = awp_adversary.calc_awp(inputs_adv=x_adv,
                                         inputs_clean=data,
                                         targets=target,
                                         beta=1,epoch=epoch,eps_adv=eps_adv)
            awp_adversary.perturb(awp)
    	# calculate robust loss
    	#Getting the  logits, mean vector and variance vector for clean image using the noise sampled for the clean image while generating adv
        logits_normal,mu_normal,logvar_normal,eps_normal = model(data,no_sample=True,epso=eps_adv)


    	#Getting the  logits, mean vector and variance vector for adv image using the noise sampled for the adv image while generating adv
        logits_adv,mu_adv,logvar_adv,eps_adv=model(x_adv,no_sample=True,epso=eps_adv)


    	#Calculate CE(adv) 
        loss_adv = F.cross_entropy(logits_adv, target)

    
    	#Calculate KLD(Adv||clean) in the latent space
    	#THe if condition belows help in stable training. So basically for first 2 iterations we assign a low weight like 0.01 to KLD loss and later this weight is increased to 1.
        if epoch<=5:
            loss_lat_adv1 = (1.0 / batch_size) *0.01*(-0.5 * torch.sum(1 + logvar_adv-logvar_normal - ((mu_normal-mu_adv).pow(2) + logvar_adv.exp())/logvar_normal.exp()))
        else:
       	    loss_lat_adv1 = (1.0 / batch_size) *1*(-0.5 * torch.sum(1 + logvar_adv-logvar_normal - ((mu_normal-mu_adv).pow(2) + logvar_adv.exp())/logvar_normal.exp()))


    	#Calculate KLD(clean||standard normal) in the latent space.
        loss_clean_lat = (1.0 / batch_size) *0.01*(-0.5 * torch.sum(1 + logvar_normal - mu_normal.pow(2) - logvar_normal.exp()))

    	#Calculate the KLD(Adv||sampled Adv) in the softmax space.
    	#Get the logits when no sampling is done
        pred_no_sample  = model(x_adv,no_sample=True,epso = torch.zeros(len(x_adv),512).cuda())[0]
        pred_sample = model(x_adv)[0]
        pred_soft_sample = nn.LogSoftmax(dim=1)(pred_sample)
        pred_soft_no_sample = nn.Softmax(dim=1)(pred_no_sample)
    	#Now calculate the loss KLD(Adv||sampled Adv) in softmax space.
        loss_soft_adv1 = 0.1*torch.nn.functional.kl_div(pred_soft_sample,pred_soft_no_sample)


    	#Returning the total loss
        loss = loss_adv  + loss_clean_lat + loss_lat_adv1 + loss_soft_adv1
        loss.backward()
        count=0
        optimizer.step()
        optimizer.zero_grad()
        tot_loss+=loss.item()
        if epoch >= args.awp_warmup:
            awp_adversary.restore(awp)
        model.train()
        # The first step of alternate iteration training ends


        #The second alternate iteration step that is normal training begins
        #Get the logits, mean vector, logvariance vector and epsilon(noise) vectors.
        pred,mu_normal,logvar_normal,eps=model(data)
        
        #Calculate the KLD(clean||standard normal) in latent space
        loss_KLD_clean=(1.0 / TRAIN_BATCH_SIZE) *0.01*(-0.5 * torch.sum(1 + logvar_normal - mu_normal.pow(2) - logvar_normal.exp()))

        #Calculate CE(clean)
        #Sampling is done to get the logits pred
        loss_CE = nn.CrossEntropyLoss()(pred, target)

        #Calculate the KLD(sample clean||no sampled clean) in the softmax space.
        #Get the logits when no sampling is done
        #no_sample=True means it will not sample epsilon in latent space but rather take is as an input which is epso.
        pred_no_sample = model(data,no_sample=True,epso = torch.zeros(len(data),512).cuda())[0]

        pred_soft_sample = nn.LogSoftmax(dim=1)(pred)
        pred_soft_no_sample = nn.Softmax(dim=1)(pred_no_sample)
        #Now calculate the loss KLD(sample clean||no sampled clean) in softmax space.
        loss_KLD2_clean = torch.nn.functional.kl_div(pred_soft_sample,pred_soft_no_sample)

        #backprop and gradient update(SGD)
        loss=loss_KLD_clean+loss_CE +loss_KLD2_clean

        loss.backward()
        optimizer.step()
        scheduler.step()
	
        #The second alternate iteration step ends

        ### Logging starts here ###
        model.eval()
        out = model(x_adv,no_sample=True,epso=torch.zeros(1,512).cuda())[0]      
        prediction = out.data.max(1)[1] 
        accuracy_pgd = prediction.eq(target.data).sum()
        epoch_accuracy_pgd+=accuracy_pgd
        model.eval()
        out = model(data,no_sample=True,epso=torch.zeros(1,512).cuda())[0]      
        prediction = out.data.max(1)[1] 
        accuracy_clean = prediction.eq(target.data).sum()
        epoch_accuracy_clean+=accuracy_clean
        model.train()
        # print progress
        if batch_idx % args.log_interval == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                       100. * batch_idx / len(train_loader), loss.item()))
            msg_train= 'Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                       100. * batch_idx / len(train_loader), loss.item())+'\n'
            log_file = open(LOSS_TRAIN,'a+')
            log_file.write(msg_train)
            log_file.close()

    msg_train_clean= 'Train Epoch: {} \t Clean_accuracy: {:.6f}'.format(
    epoch, 100*epoch_accuracy_clean.item()/(128*len(train_loader)))+'\n'
    log_file = open(ACC_CLEAN_TRAIN,'a+')
    log_file.write(msg_train_clean)
    log_file.close()

    msg_train_pgd= 'Train Epoch: {} \t PGD_10_accuracy: {:.6f}'.format(
    epoch, 100*epoch_accuracy_pgd.item()/(128*len(train_loader)))+'\n'
    log_file = open(ACC_PGD10_TRAIN,'a+')
    log_file.write(msg_train_pgd)
    log_file.close()
    end_time=time.time()
    print("Time for epoch:",end_time - start_time)
##################################




def main():
    
    model = VAE().to(device)
    model = nn.DataParallel(model)
    lr_steps = args.epochs * len(train_loader)
    optimizer = optim.SGD(model.parameters(), lr=args.lr,weight_decay=args.weight_decay)
    proxy = VAE().cuda()
    proxy = nn.DataParallel(proxy).to(device)
    proxy_optim = optim.SGD(proxy.parameters(), lr=args.lr)
    awp_adversary = TradesAWP(model=model, proxy=proxy, proxy_optim=proxy_optim, gamma=args.awp_gamma)
    scheduler = torch.optim.lr_scheduler.CyclicLR(optimizer, base_lr=0.0, max_lr=0.1, step_size_up=lr_steps / 2, step_size_down=lr_steps / 2, gamma=3)

    for epoch in range(1, args.epochs + 1):
        lr = scheduler.get_lr()[0]
        print("Current learning rate is",lr)
        # adversarial training
        train(args, model, device, train_loader, optimizer ,epoch,scheduler,awp_adversary)
        
        torch.save(model.state_dict(),
                       os.path.join(model_dir, 'model-res18FC-epoch{}.pt'.format(epoch)))
        torch.save(optimizer.state_dict(),
                       os.path.join(model_dir, 'opt-res18FC-checkpoint_epoch{}.tar'.format(epoch)))


if __name__ == '__main__':
    main()



######## Evaluations of trained models on Val Set Begins ###########

def PGD(model,loss,data,target,eps=0.1,eps_iter=0.1,bounds=[],steps=1):
    if model.training:
        assert 'Model is in  training mode'
    tar = Variable(target.cuda())
    data = data.cuda()
    B,C,H,W = data.size()
    noise  = torch.FloatTensor(np.random.uniform(-eps,eps,(B,C,H,W))).cuda()
    noise  = torch.clamp(noise,-eps,eps)
    for step in range(steps):
        # convert data and corresponding into cuda variable
        img = data + noise
        img = Variable(img,requires_grad=True)
        # make gradient of img to zeros
        zero_gradients(img) 
        # forward pass
        out  = model(img,no_sample=True,epso=torch.zeros(1,512).cuda())[0]
        #compute loss using true label
        cost = loss(out,tar)
        #backward pass
        cost.backward()
        #get gradient of loss wrt data
        per =  torch.sign(img.grad.data)
        #convert eps 0-1 range to per channel range 
        per[:,0,:,:] = (eps_iter * (bounds[0,1] - bounds[0,0])) * per[:,0,:,:]
        if(per.size(1)>1):
            per[:,1,:,:] = (eps_iter * (bounds[1,1] - bounds[1,0])) * per[:,1,:,:]
            per[:,2,:,:] = (eps_iter * (bounds[2,1] - bounds[2,0])) * per[:,2,:,:]
        #  ascent
        adv = img.data + per.cuda()
        #clip per channel data out of the range
        img.requires_grad =False
        img[:,0,:,:] = torch.clamp(adv[:,0,:,:],bounds[0,0],bounds[0,1])
        if(per.size(1)>1):
            img[:,1,:,:] = torch.clamp(adv[:,1,:,:],bounds[1,0],bounds[1,1])
            img[:,2,:,:] = torch.clamp(adv[:,2,:,:],bounds[2,0],bounds[2,1])
        img = img.data
        noise = img - data
        noise  = torch.clamp(noise,-eps,eps)
    img = data + noise
    return img


if not os.path.exists(model_dir+"/results"):
    os.makedirs(model_dir+"/results")

model = VAE().to(device)
model = nn.DataParallel(model)
model.eval()

########################## FIND BEST MODEL###############################################

ACC_EPOCH_LOG_NAME = model_dir+'/results/clean_acc.txt'
ACC_IFGSM_EPOCH_LOG_NAME = model_dir+'/results/pgd7_acc_D100.txt'

accuracy_log = np.zeros(args.epochs)
for epoch in range(0,args.epochs):
    model_name = model_dir+'/model-res18FC-epoch'+str(epoch+1)+'.pt'
    model.load_state_dict(torch.load(model_name))
    model.eval()
    eps=8.0/255
    accuracy = 0
    accuracy_ifgsm = 0
    i = 0
    for data, target in val_loader:
        data   = Variable(data).cuda()
        target = Variable(target).cuda()
        out = model(data,no_sample=True,epso=torch.zeros(1,512).cuda())[0]
        prediction = out.data.max(1)[1] 
        accuracy = accuracy + prediction.eq(target.data).sum()
        i = i + 1 
    for data, target in val_loader:
        data = PGD(model,loss,data,target,eps=8.0/255,eps_iter=2.0/255,bounds=np.array([[0,1],[0,1],[0,1]]),steps=7)
        data   = Variable(data).cuda()
        target = Variable(target).cuda()
        out = model(data,no_sample=True,epso=torch.zeros(1,512).cuda())[0]
        prediction = out.data.max(1)[1] 
        accuracy_ifgsm = accuracy_ifgsm + prediction.eq(target.data).sum()
    acc = (accuracy.item()*1.0) / (i*VAL_BATCH_SIZE) * 100
    acc_ifgsm = (accuracy_ifgsm.item()*1.0) / (i*VAL_BATCH_SIZE) * 100
    #log accuracy to file
    msg= str(epoch+1)+','+str(acc)+'\n'
    log_file = open(ACC_EPOCH_LOG_NAME,'a+')
    log_file.write(msg)
    log_file.close()
    
    msg1= str(epoch+1)+','+str(acc_ifgsm)+'\n'
    log_file = open(ACC_IFGSM_EPOCH_LOG_NAME,'a+')
    log_file.write(msg1)
    log_file.close()

    accuracy_log[epoch] = acc_ifgsm

    sys.stdout.write('\r')
    sys.stdout.write('| Epoch [%3d/%3d] : Acc:%f \t\t'
            %(epoch+1, args.epochs,acc))
    sys.stdout.flush()  

log_file = open(ACC_IFGSM_EPOCH_LOG_NAME,'a+')
msg = '\nEpoch,'+str(accuracy_log.argmax()+1)+',Acc,'+str(accuracy_log.max())+'\n'
log_file.write(msg)
log_file.close()



model_name = model_dir+'/model-res18FC-epoch'+str(accuracy_log.argmax()+1)+'.pt'
model.load_state_dict(torch.load(model_name))
best_model_name = model_dir+'/results/best_model-res18FC.pt'
torch.save(model.state_dict(),best_model_name)



