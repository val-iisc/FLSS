import torch
from collections import OrderedDict
import torch.nn as nn
import torch.nn.functional as F
EPS = 1E-20


def diff_in_weights(model, proxy):
    diff_dict = OrderedDict()
    model_state_dict = model.state_dict()
    proxy_state_dict = proxy.state_dict()
    for (old_k, old_w), (new_k, new_w) in zip(model_state_dict.items(), proxy_state_dict.items()):
        if len(old_w.size()) <= 1:
            continue
        if 'weight' in old_k:
            diff_w = new_w - old_w
            diff_dict[old_k] = old_w.norm() / (diff_w.norm() + EPS) * diff_w
    return diff_dict


def add_into_weights(model, diff, coeff=1.0):
    names_in_diff = diff.keys()
    with torch.no_grad():
        for name, param in model.named_parameters():
            if name in names_in_diff:
                param.add_(coeff * diff[name])


class TradesAWP(object):
    def __init__(self, model, proxy, proxy_optim, gamma):
        super(TradesAWP, self).__init__()
        self.model = model
        self.proxy = proxy
        self.proxy_optim = proxy_optim
        self.gamma = gamma

    def calc_awp(self, inputs_adv, inputs_clean, targets, beta,epoch,eps_adv):
        self.proxy.load_state_dict(self.model.state_dict())
        self.proxy.train()

        #loss_natural = F.cross_entropy(self.proxy(inputs_clean), targets)
        #loss_robust = F.kl_div(F.log_softmax(self.proxy(inputs_adv), dim=1),
        #                       F.softmax(self.proxy(inputs_clean), dim=1),
        #                       reduction='batchmean')

        
        batch_size = 128
        logits_normal,mu_normal,logvar_normal,eps_normal = self.proxy(inputs_clean,no_sample=True,epso=eps_adv)


    	#Getting the  logits, mean vector and variance vector for adv1 image using the noise sampled for the adv1 image while generating adv1
        logits_adv,mu_adv,logvar_adv,eps_adv=self.proxy(inputs_adv,no_sample=True,epso=eps_adv)


        loss_adv = F.cross_entropy(logits_adv, targets)

    
    	#Calculate KLD(Adv1||clean) in the latent space
    	#THe if condition belows help in stable training. So basically for first 2 iterations we assign a low weight like 0.01 to KLD loss and later this weight is increased to 1.
        if epoch<=5:
            loss_lat_adv1 = (1.0 / batch_size) *0.01*(-0.5 * torch.sum(1 + logvar_adv-logvar_normal - ((mu_normal-mu_adv).pow(2) + logvar_adv.exp())/logvar_normal.exp()))
        else:
            loss_lat_adv1 = (1.0 / batch_size) *1*(-0.5 * torch.sum(1 + logvar_adv-logvar_normal - ((mu_normal-mu_adv).pow(2) + logvar_adv.exp())/logvar_normal.exp()))


    	#Calculate KLD(clean||standard normal) in the latent space.
    	# Earlier assigned a weight of 0.01 to KLD loss   
        loss_clean_lat = (1.0 / batch_size) *0.01*(-0.5 * torch.sum(1 + logvar_normal - mu_normal.pow(2) - logvar_normal.exp()))

    	#Calculate the KLD(sampled Adv1||Adv) in the softmax space.
    	#Get the logits when no sampling is done
        pred_no_sample  = self.proxy(inputs_adv,no_sample=True,epso = torch.zeros(len(inputs_adv),512).cuda())[0]
        pred_sample = self.proxy(inputs_adv)[0]
        pred_soft_sample = nn.LogSoftmax(dim=1)(pred_sample)
        pred_soft_no_sample = nn.Softmax(dim=1)(pred_no_sample)
    	#Now calculate the loss KLD(Adv1||sampled Adv1) in softmax space.
        loss_soft_adv1 = 0.1*torch.nn.functional.kl_div(pred_soft_sample,pred_soft_no_sample)


    	#Returning the total loss
        loss = loss_adv  + loss_clean_lat + loss_lat_adv1 + loss_soft_adv1

        loss = - 1.0 * loss

        self.proxy_optim.zero_grad()
        loss.backward()
        self.proxy_optim.step()

        # the adversary weight perturb
        diff = diff_in_weights(self.model, self.proxy)
        return diff

    def perturb(self, diff):
        add_into_weights(self.model, diff, coeff=1.0 * self.gamma)

    def restore(self, diff):
        add_into_weights(self.model, diff, coeff=-1.0 * self.gamma)




