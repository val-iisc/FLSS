  
'''ResNet in PyTorch.
Added an additional class VAE and a one cross one convolutional layer to predict mean and standard deviation and perform sampling in the last layer of resnet18 before passing the feature maps to FC layers.
Reference:
[1] Kaiming He, Xiangyu Zhang, Shaoqing Ren, Jian Sun
    Deep Residual Learning for Image Recognition. arXiv:1512.03385
'''
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion*planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion*planes, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion*planes)
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out

class ResNet(nn.Module):
    def __init__(self, block, num_blocks, num_classes=10):
        super(ResNet, self).__init__()
        self.in_planes = 64

        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.layer1 = self._make_layer(block, 64, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, 128, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, 256, num_blocks[2], stride=2)
        self.layer4 = self._make_layer(block, 512, num_blocks[3], stride=2)
        self.conv_mu = nn.Conv2d(512, 512, kernel_size=1, stride=1, padding=0)
        self.conv_logvar = nn.Conv2d(512, 512, kernel_size=1, stride=1, padding=0)

    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1]*(num_blocks-1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out_mu = self.conv_mu(out)
        out_logvar = self.conv_logvar(out)
        out_mu = F.avg_pool2d(out_mu, 4)
        out_mu = out_mu.view(out_mu.size(0), -1)
        out_logvar = F.avg_pool2d(out_logvar, 4)
        out_logvar = out_logvar.view(out_logvar.size(0), -1)
        return out_mu,out_logvar

def ResNet18():
    return ResNet(BasicBlock, [2,2,2,2])

class VAE(nn.Module):
    def __init__(self,sample_std):
        super(VAE, self).__init__()

        self.encode = ResNet18().cuda()
        self.sample_std = sample_std
        self.relu = nn.ReLU()
        #Classifier_encoder
        #self.fc1_encoder = nn.Linear(512,128)
        #self.fc1_bn1_encoder = nn.BatchNorm1d(128)
        self.fc1_encoder = nn.Linear(512,100)

        

    def reparameterize(self, mu, logvar,noi=0,noi_sample=1):
        
        std = logvar.mul(0.5).exp()
        if noi_sample ==1:
            if noi==0:
                eps = self.sample_std*torch.zeros(1,512).cuda()
        else:
            eps = self.sample_std*noi
        return eps.mul(std).add(mu),eps

    def classifier_encoder(self,z):
        #out= self.relu(self.fc1_bn1_encoder(self.fc1_encoder(z)))
        out= self.fc1_encoder(z)
        return out


    #to edit and need to apply PGD
    def forward(self, x,noi=0,noi_sample=1):    
        mu, logvar = self.encode(x)
        z,eps = self.reparameterize(mu, logvar,noi,noi_sample)
        return self.classifier_encoder(z)






