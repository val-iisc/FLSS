import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import numpy as np
import argparse
import time
import torch.optim as optim

from torch.autograd.gradcheck import zero_gradients

def max_margin_logit_loss(logits,y):
    logit_org = logits.gather(1,y.view(-1,1))
    logit_target = logits.gather(1,(logits - torch.eye(10)[y].to("cuda") * 9999).argmax(1, keepdim=True))
    loss = -logit_org + logit_target
    loss = torch.mean(loss)
    return loss

def MT(model,data,target,eps=0.1,eps_iter=0.1,bounds=[],steps=100,w_reg=25,lin=50,SCHED=[],drop=1,multi_tar=1):
    """
    model
    loss : loss used for training
    data : input to network
    target : ground truth label corresponding to data
    eps : perturbation srength added to image
    eps_iter
    """
    #Raise error if in training mode
    if model.training:
        assert 'Model is in  training mode'
    tar = Variable(target.cuda())
    data = data.cuda()
    B,C,H,W = data.size()
    noise  = torch.FloatTensor(np.random.uniform(-eps,eps,(B,C,H,W))).cuda()
    #noise  = eps*torch.sign(noise)
    
    new_tar = (tar + multi_tar)%10
    
    img_arr = []
    W_REG = w_reg
    orig_img = data+noise
    orig_img = Variable(orig_img,requires_grad=True)
    for step in range(steps):
        # convert data and corresponding into cuda variable
        img = data + noise
        img = Variable(img,requires_grad=True)
        
        if step in SCHED:
            eps_iter /= drop
        
        # make gradient of img to zeros
        zero_gradients(img) 
        # forward pass        
        #orig_out = model(orig_img)
        #P_out = nn.Softmax(dim=1)(orig_out)
        
        
        out  = model(img)
        #Q_out = nn.Softmax(dim=1)(out)
        #compute loss using true label
        if multi_tar==10:
            cost =  max_margin_logit_loss(out,tar)
        else:
            cost = (-out[range(len(out)),tar] + out[range(len(out)),new_tar]).mean(0)
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
    return data + noise



def GAMA_MT(model,data,target,eps,eps_iter,bounds,steps,w_reg,lin,SCHED,drop,rr,new_targ):
    """
    model
    loss : loss used for training
    data : input to network
    target : ground truth label corresponding to data
    eps : perturbation srength added to image
    eps_iter
    """
    #Raise error if in training mode
    if model.training:
        assert 'Model is in  training mode'
    tar = Variable(target.cuda())
    data = data.cuda()
    B,C,H,W = data.size()
    noise  = torch.FloatTensor(np.random.uniform(-eps,eps,(B,C,H,W))).cuda()
    noise  = eps*torch.sign(noise)
    img_arr = []
    W_REG = w_reg
    orig_img = data+noise
    orig_img = Variable(orig_img,requires_grad=True)
    for step in range(steps):
        # convert data and corresponding into cuda variable
        img = data + noise
        img = Variable(img,requires_grad=True)
        
        if step in SCHED:
            eps_iter /= drop
        
        # make gradient of img to zeros
        zero_gradients(img) 
        # forward pass        
        orig_out = model(orig_img)
        P_out = nn.Softmax(dim=1)(orig_out)
        
        out  = model(img)
        Q_out = nn.Softmax(dim=1)(out)
        #compute loss using true label
        if rr==5:
            if step <= lin:
                cost =  W_REG*((P_out - Q_out)**2.0).sum(1).mean(0) + max_margin_loss(Q_out,tar)
                W_REG -= w_reg/lin
            else:
                cost = max_margin_loss(Q_out,tar)
        else:
            if step <= lin:
                cost =  W_REG*((P_out - Q_out)**2.0).sum(1).mean(0) + (-Q_out[range(len(Q_out)),tar] + Q_out[range(len(Q_out)),new_targ]).mean(0)
                W_REG -= w_reg/lin
            else:
                cost = (-Q_out[range(len(Q_out)),tar] + Q_out[range(len(Q_out)),new_targ]).mean(0)
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

    return data + noise

def max_margin_loss(x,y):
    B = y.size(0)
    corr = x[range(B),y]

    x_new = x - 1000*torch.eye(10)[y].cuda()
    tar = x[range(B),x_new.argmax(dim=1)]
    loss = tar - corr
    loss = torch.mean(loss)
    
    return loss

def GAMA_PGD(model,data,target,eps,eps_iter,bounds,steps,w_reg,lin,SCHED,drop):
    """
    model
    loss : loss used for training
    data : input to network
    target : ground truth label corresponding to data
    eps : perturbation srength added to image
    eps_iter
    """
    #Raise error if in training mode
    if model.training:
        assert 'Model is in  training mode'
    tar = Variable(target.cuda())
    data = data.cuda()
    B,C,H,W = data.size()
    noise  = torch.FloatTensor(np.random.uniform(-eps,eps,(B,C,H,W))).cuda()
    noise  = eps*torch.sign(noise)
    img_arr = []
    W_REG = w_reg
    orig_img = data+noise
    orig_img = Variable(orig_img,requires_grad=True)
    for step in range(steps):
        # convert data and corresponding into cuda variable
        img = data + noise
        img = Variable(img,requires_grad=True)
        
        if step in SCHED:
            eps_iter /= drop
        
        # make gradient of img to zeros
        zero_gradients(img) 
        # forward pass        
        orig_out = model(orig_img)
        P_out = nn.Softmax(dim=1)(orig_out)
        
        out  = model(img)
        Q_out = nn.Softmax(dim=1)(out)
        #compute loss using true label
        if step <= lin:
            cost =  W_REG*((P_out - Q_out)**2.0).sum(1).mean(0) + max_margin_loss(Q_out,tar)
            W_REG -= w_reg/lin
        else:
            cost = max_margin_loss(Q_out,tar)
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

    return data + noise
    
class Attack():
    def __init__(self, model,eot_iter, norm='Linf', eps=.3, restarts=5, seed=None, verbose=True,
                 attacks_to_run=['apgd-ce','apgd-dlr','fab','square','MM'],
                 plus=False, is_tf_model=False, device='cuda'): #
        self.model = model
        self.norm = norm
        self.eot_iter = eot_iter
        assert norm in ['Linf', 'L2']
        self.epsilon = eps
        self.restarts = restarts
        self.seed = seed
        self.verbose = verbose
        if plus:
            attacks_to_run.extend(['apgd-t', 'fab-t'])
        self.attacks_to_run = attacks_to_run
        self.plus = plus
        self.is_tf_model = is_tf_model
        self.device = device

        from autopgd_pt import APGDAttack
        self.apgd = APGDAttack(self.model, n_restarts=self.restarts, n_iter=100, verbose=False,
            eps=self.epsilon, norm=self.norm, eot_iter=1, rho=.75, seed=self.seed, device=self.device)
            
        from fab_pt import FABAttack
        self.fab = FABAttack(self.model, n_restarts=self.restarts, n_iter=100, eps=self.epsilon, seed=self.seed,
            norm=self.norm, verbose=False, device=self.device)
        
        from square import SquareAttack
        self.square = SquareAttack(self.model, p_init=.8, n_queries=5000, eps=self.epsilon, norm=self.norm,
            n_restarts=1, seed=self.seed, verbose=False, device=self.device, resc_schedule=False)
                
        from autopgd_pt import APGDAttack_targeted
        self.apgd_targeted = APGDAttack_targeted(self.model, n_restarts=1, n_iter=100, verbose=False,
            eps=self.epsilon, norm=self.norm, eot_iter=1, rho=.75, seed=self.seed, device=self.device)
    
    
    def get_logits(self, x):
        return self.model(x)

    def _pgd_whitebox(self,
                      X,
                      y,
                      epsilon=0.03137254,
                      num_steps=100,
                      step_size=0.007843137,eot_iter=1):
        out = 	self.model(X)
        err = (out.data.max(1)[1] != y.data).float().sum()
        X_pgd = Variable(X.data, requires_grad=True)
        
        random_noise = torch.FloatTensor(*X_pgd.shape).uniform_(-epsilon, epsilon).cuda()
        X_pgd = Variable(X_pgd.data + random_noise, requires_grad=True)
        for _ in range(num_steps):
            summer_grad = torch.zeros(X_pgd.shape).cuda()
            for j in range(eot_iter):
                opt = optim.SGD([X_pgd], lr=1e-3)
                opt.zero_grad()
                with torch.enable_grad():
                    loss = nn.CrossEntropyLoss()(self.model(X_pgd), y)
                loss.backward()
                summer_grad = summer_grad+X_pgd.grad.data
            eta = step_size * ((summer_grad).sign())
            X_pgd = Variable(X_pgd.data + eta, requires_grad=True)
            eta = torch.clamp(X_pgd.data - X.data, -epsilon, epsilon)
            X_pgd = Variable(X.data + eta, requires_grad=True)
            X_pgd = Variable(torch.clamp(X_pgd, 0, 1.0), requires_grad=True)
        return X_pgd


    def get_seed(self):
        return time.time() if self.seed is None else self.seed

    def run_standard_evaluation(self, x_orig, y_orig, lister_attack,threshold,bs=1000,type_of_thresholder="original",RR=10):
        
        with torch.no_grad():
            # calculate accuracy
            n_batches = int(np.ceil(x_orig.shape[0] / bs))
            robust_flags = torch.zeros(x_orig.shape[0], dtype=torch.bool, device=x_orig.device)
            #print("Robust_flag",len(robust_flags))
            for batch_idx in range(n_batches):
                start_idx = batch_idx * bs
                end_idx = min( (batch_idx + 1) * bs, x_orig.shape[0])

                x = x_orig[start_idx:end_idx, :].clone().to(self.device)
                y = y_orig[start_idx:end_idx].clone().to(self.device)
                correct_batch = y.eq(y)
                robust_flags[start_idx:end_idx] = correct_batch.detach().to(robust_flags.device)
            #print("Robust_flag",len(robust_flags))
            x_adv = x_orig.clone().detach()
            startt = time.time()
            attack = lister_attack[0]

            index_array_accepted_false_0 =[]
            index_array_accepted_true_0 = []  
            index_array_accepted_false =[]
            index_array_accepted_true = []
            index_array_rejected=[]

            num_robust = torch.sum(robust_flags).item()
            n_batches = int(np.ceil(num_robust / bs))
            robust_lin_idcs = torch.nonzero(robust_flags, as_tuple=False)
            if num_robust > 1:
                robust_lin_idcs.squeeze_()
            noise_mat = torch.Tensor(np.load('./mat.npy')).cuda()
            #print("Number of batches are:",n_batches)  
            for batch_idx in range(n_batches):
                    start_idx = batch_idx * bs
                    end_idx = min((batch_idx + 1) * bs, num_robust)

                    batch_datapoint_idcs = robust_lin_idcs[start_idx:end_idx]
                    if len(batch_datapoint_idcs.shape) > 1:
                        batch_datapoint_idcs.squeeze_(-1)
                    x = x_orig[batch_datapoint_idcs, :].clone().to(self.device)
                    y = y_orig[batch_datapoint_idcs].clone().to(self.device)

                    # make sure that x is a 4d tensor even if there is only a single datapoint left
                    if len(x.shape) == 3:
                        x.unsqueeze_(dim=0)
                    
                    # run attack
                    if attack == 'apgd_ce':
                        # apgd on cross-entropy loss
                        print("running apgd_ce")
                        self.apgd.loss = 'ce'
                        self.apgd.seed = self.get_seed()
                        _, adv_curr = self.apgd.perturb(x, y, cheap=True)
                    
                    elif attack == 'apgd_dlr':
                        print("running apgd_dlr")
                        # apgd on dlr loss
                        self.apgd.loss = 'dlr'
                        self.apgd.seed = self.get_seed()
                        _, adv_curr = self.apgd.perturb(x, y, cheap=True)
                    
                    elif attack == 'fab':
                        print("running fab")
                        # fab
                        self.fab.targeted = False
                        self.fab.seed = self.get_seed()
                        adv_curr = self.fab.perturb(x, y)
                    
                    elif attack == 'square':
                        # square
                        print("running square")
                        self.square.seed = self.get_seed()
                        adv_curr = self.square.perturb(x, y)

                    elif attack == 'clean':
                        # square
                        print("running clean")
                        adv_curr = x

                    elif attack == 'Gama_pgd':
                        print("running Gama_pgd")
                        with torch.enable_grad():
                           adv_curr = GAMA_PGD(self.model,x,y,eps=self.epsilon,eps_iter=2*self.epsilon,bounds=np.array([[0,1],[0,1],[0,1]]),steps=100,w_reg=50,lin=25,SCHED=[60,85],drop=10) 
                        adv_curr = Variable(adv_curr).cuda() 

                    elif attack == 'Gama_MT':
                        #Max margin attack
                        print("running GAMA_MT")
                        with torch.enable_grad():
                           out_clean = self.model(x)
                           #print(out_clean)
                           topk = torch.topk(out_clean,5+1)[1]
                           #print(topk)
                           adv_curr = GAMA_MT(self.model,x,y,eps=self.epsilon,eps_iter=2*self.epsilon,bounds=np.array([[0,1],[0,1], [0,1]]),steps=100,w_reg=50,lin=25,SCHED=[60,85],drop=10,rr=RR+1,new_targ=topk[range(len(y)),RR+1]) 

                    elif attack == 'MM':
                        #Max margin attack
                        print("running MM")
                        with torch.enable_grad():
                           adv_curr = MT(self.model,x,y,eps=self.epsilon,eps_iter=2*self.epsilon,bounds=np.array([[0,1],[0,1],[0,1]]),steps=100,w_reg=0,lin=0,SCHED=[50,75],drop=10,multi_tar=10)
                   
                    elif attack == 'MT':
                        print("running MT")
                        # apgd on dlr loss
                        #self.MT.loss = 'MT'
                        #self.apgd.seed = self.get_seed()
                        with torch.enable_grad():
                           adv_curr = MT(self.model,x,y,eps=self.epsilon,eps_iter=2*self.epsilon,bounds=np.array([[0,1],[0,1],[0,1]]),steps=100,w_reg=0,lin=0,SCHED=[50,75],drop=10,multi_tar=RR+1)

                    elif attack == 'pgd':
                        #pgd
                        print("running pgd")
                        adv_curr = self._pgd_whitebox( x, y,eot_iter=self.eot_iter,epsilon=self.epsilon,step_size=self.epsilon/4)
                    else:
                        raise ValueError('Attack not supported')
                    
                    maxa=np.zeros(len(adv_curr))
                    output = np.zeros(len(adv_curr))
                    lst_ind=np.zeros([len(adv_curr),10])

                    for i in range(100):
                        noise = noise_mat[i]
                        answer = self.model(adv_curr,noi=noise,noi_sample=0)

                        ans=answer.max(dim=1)[1]
                        for j in range(len(ans)):
                            lst_ind[j][ans[j]]+=1
                            if lst_ind[j][ans[j]]>maxa[j]:
                               maxa[j] = lst_ind[j][ans[j]]
                               output[j] = ans[j]
                    y_label = y.detach().cpu().long().numpy()
                    
                    for i in range(len(ans)):
                        # Finding the 0% correct and incorrect samples
                        if output[i]==y_label[i]:    
                            index_array_accepted_true_0.append((batch_idx*bs+i))
                        else: 
                            index_array_accepted_false_0.append((batch_idx*bs+i))
                        #Running the original rejection 
                        if type_of_thresholder=="original":
                            if maxa[i]<=int(threshold):
                                index_array_rejected.append((batch_idx*bs+i))
                            else:
                                if output[i]==y_label[i]:    
                                    index_array_accepted_true.append((batch_idx*bs+i))
                                else: 
                                    index_array_accepted_false.append((batch_idx*bs+i))

                    false_batch = ~y.eq(torch.Tensor(output).cuda()).to(robust_flags.device)
                    non_robust_lin_idcs = batch_datapoint_idcs[false_batch]
                    robust_flags[non_robust_lin_idcs] = False

                    x_adv[non_robust_lin_idcs] = adv_curr[false_batch].detach().to(x_adv.device)
                
                    if self.verbose:
                        num_non_robust_batch = torch.sum(false_batch)    
                        print('{} - {}/{} - {} out of {} successfully perturbed'.format(
                            attack, batch_idx + 1, n_batches, num_non_robust_batch, x.shape[0]))
        return np.array(index_array_accepted_true_0), np.array(index_array_accepted_false_0),np.array(index_array_rejected), np.array(index_array_accepted_true), np.array(index_array_accepted_false)
        
        
    def cheap(self):
        self.apgd.n_restarts = 1
        self.fab.n_restarts = 1
        self.apgd_targeted.n_restarts = 1
        self.square.n_queries = 1000
        self.square.resc_schedule = True
        self.plus = False



