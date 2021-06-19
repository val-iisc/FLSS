import os
import argparse  
import torch
import torch.nn as nn
import torchvision.datasets as datasets
import torch.utils.data as data
import torchvision.transforms as transforms
import numpy as np
import sys
from resnet import VAE
sys.path.insert(0,'..')

import torchvision.datasets as datasets
import torch.utils.data as data
import torchvision.transforms as transforms
import torchvision  
import numpy as np
parser = argparse.ArgumentParser()
parser.add_argument('--data_dir', type=str, default='../../data')
parser.add_argument('--norm', type=str, default='Linf')
parser.add_argument('--epsilon', type=float, default=8/255.)
parser.add_argument('--model', type=str, default='../../../FLSS_cifar10.pt')
parser.add_argument('--n_ex', type=int, default=10000)
parser.add_argument('--restarts', type=int, default=1)
parser.add_argument('--individual', action='store_true')
parser.add_argument('--cheap', action='store_true')
parser.add_argument('--save_dir', type=str, default='./results')
parser.add_argument('--batch_size', type=int, default=250)
parser.add_argument('--sample_std', type=int, default=2)
parser.add_argument('--plus', action='store_true')
parser.add_argument('--eot', type=int, default=1)
parser.add_argument('--attacks_to_run', type=str, nargs='*', default=['clean','apgd_ce','apgd_dlr','MM','Gama_pgd','pgd','Gama_MT'])
parser.add_argument('--balanced_test_size', type=int, default=1000)
parser.add_argument('--threshold', type=float, default=0)
parser.add_argument('--take_threshold_from_file', type=bool, default=True)
parser.add_argument('--type_of_thresholder', type=str, default="original")
parser.add_argument('--full_test_eval', type=bool, default=True)
parser.add_argument('--no_state_dict', type=bool, default=True)
parser.add_argument('--prefix_save_name', type=str, default="index_array")
args = parser.parse_args()
if __name__ == '__main__':

    # load model
    sys.path.append("./")
    model  = VAE(args.sample_std).cuda()
    model = torch.nn.DataParallel(model)
    if not args.no_state_dict:
       model.load_state_dict(torch.load(args.model)["state_dict"])
    else:
       model.load_state_dict(torch.load(args.model))
    #model = torch.nn.DataParallel(model)
    model.eval()
    model.cuda()

    # load data
    transform_test = transforms.Compose([
                transforms.ToTensor(),])
    val_set    = torchvision.datasets.CIFAR10(root=args.data_dir, train=False , download=True, transform=transform_test)
    test_set_orig   = torchvision.datasets.CIFAR10(root=args.data_dir, train=False, download=True, transform=transform_test)

    test_indices = list(range(10000))
    val_indices = []
    count = np.zeros(10)
    if not args.full_test_eval:
        for index in range(len(test_set_orig)):
            _, target = test_set_orig[index]
            if(np.all(count==100)):
                break
            if(count[target]<100):
                count[target] += 1
                val_indices.append(index)
                test_indices.remove(index)
        
        test_set = torch.utils.data.Subset(test_set_orig,val_indices)   
        print("Overlap indices:",list(set(test_indices) & set(val_indices)))
        print('CIFAR10 dataloader balanced {} Test Images: Done'.format(len(test_set)))
         
    else:
        test_set = test_set_orig
        print('CIFAR10 dataloader 10K Test Images: Done') 
    test_loader = torch.utils.data.DataLoader(test_set,shuffle=False,batch_size=250)

    
    # create save dir
    if not os.path.exists(args.save_dir):
        os.makedirs(args.save_dir)
    if not os.path.exists("attack_arrays"):
        os.makedirs("attack_arrays")
    if not os.path.exists("attack_arrays_0%"):
        os.makedirs("attack_arrays_0%")
    # load attack    

    from attack import Attack
    sys.path.append("./attack-master")
    adversary = Attack(model, eot_iter=args.eot,norm=args.norm, eps=args.epsilon, restarts = args.restarts, attacks_to_run=args.attacks_to_run)
    
    l = [x for (x, y) in test_loader]
    x_test = torch.cat(l, 0)
    l = [y for (x, y) in test_loader]
    y_test = torch.cat(l, 0)
    
    # cheap version
    if args.cheap:
        adversary.cheap()
    
    # plus version
    if args.plus:
        adversary.plus = True
    if args.take_threshold_from_file:
        THRESHOLD = '../threshold.txt'
        log_file = open(THRESHOLD,'r')
        mssg = int(log_file.read())
        print("Taking the threshold value from the threshold file")
        print("The threshold value is:",int(mssg)+1)
        args.threshold = int(mssg)+1

    with torch.no_grad():
        if not args.individual:

            print("The rejection scheme used is:",args.type_of_thresholder)
            for attack in args.attacks_to_run:
                if attack=='Gama_MT':
                    for r in range(5):
                        index_array_accepted_true_0,index_array_accepted_false_0,index_array_rejected,index_array_accepted_true,index_array_accepted_false = adversary.run_standard_evaluation(x_test, y_test, lister_attack=[attack],threshold=args.threshold,bs=args.batch_size,type_of_thresholder=args.type_of_thresholder,RR=r)
                        np.save('./attack_arrays/{}_{}_rejected{}.npy'.format(args.prefix_save_name,attack,r),index_array_rejected)
                        np.save('./attack_arrays/{}_{}_accepted_true{}.npy'.format(args.prefix_save_name,attack,r),index_array_accepted_true)
                        np.save('./attack_arrays/{}_{}_accepted_false{}.npy'.format(args.prefix_save_name,attack,r),index_array_accepted_false)
                        np.save('./attack_arrays_0%/{}_{}_accepted_true{}.npy'.format(args.prefix_save_name,attack,r),index_array_accepted_true_0)
                        np.save('./attack_arrays_0%/{}_{}_accepted_false{}.npy'.format(args.prefix_save_name,attack,r),index_array_accepted_false_0)
                        print("The number of images accepted and correctly classified 0% for {}{} is {}".format(attack,r,len(index_array_accepted_true_0)))
                        print("The number of images accepted and misclassified 0% for {}{} is{}".format(attack,r,len(index_array_accepted_false_0)))
                        print("The number of images accepted and correctly classified 10% for {}{} is {}".format(attack,r,len(index_array_accepted_true)))
                        print("The number of images accepted and misclassified 10% for {}{} is{}".format(attack,r,len(index_array_accepted_false)))
                        print("The number of images rejected 10% for {}{} is{}".format(attack,r,len(index_array_rejected)))

                elif attack=='MT':
                    for r in range(10):
                        index_array_accepted_true_0,index_array_accepted_false_0,index_array_rejected,index_array_accepted_true,index_array_accepted_false = adversary.run_standard_evaluation(x_test, y_test, lister_attack=[attack],threshold=args.threshold,bs=args.batch_size,type_of_thresholder=args.type_of_thresholder,RR=r)
                        np.save('./attack_arrays/{}_{}_rejected{}.npy'.format(args.prefix_save_name,attack,r),index_array_rejected)
                        np.save('./attack_arrays/{}_{}_accepted_true{}.npy'.format(args.prefix_save_name,attack,r),index_array_accepted_true)
                        np.save('./attack_arrays/{}_{}_accepted_false{}.npy'.format(args.prefix_save_name,attack,r),index_array_accepted_false)
                        np.save('./attack_arrays_0%/{}_{}_accepted_true{}.npy'.format(args.prefix_save_name,attack,r),index_array_accepted_true_0)
                        np.save('./attack_arrays_0%/{}_{}_accepted_false{}.npy'.format(args.prefix_save_name,attack,r),index_array_accepted_false_0)
                        print("The number of images accepted and correctly classified 0% for {}{} is {}".format(attack,r,len(index_array_accepted_true_0)))
                        print("The number of images accepted and misclassified 0% for {}{} is{}".format(attack,r,len(index_array_accepted_false_0)))
                        print("The number of images accepted and correctly classified 10% for {}{} is {}".format(attack,r,len(index_array_accepted_true)))
                        print("The number of images accepted and misclassified 10% for {}{} is{}".format(attack,r,len(index_array_accepted_false)))
                        print("The number of images rejected 10% for {}{} is{}".format(attack,r,len(index_array_rejected)))

                else:
                    index_array_accepted_true_0,index_array_accepted_false_0,index_array_rejected,index_array_accepted_true,index_array_accepted_false = adversary.run_standard_evaluation(x_test, y_test, lister_attack=[attack],threshold=args.threshold,bs=args.batch_size,type_of_thresholder=args.type_of_thresholder)
                    np.save('./attack_arrays/{}_{}_rejected.npy'.format(args.prefix_save_name,attack),index_array_rejected)
                    np.save('./attack_arrays/{}_{}_accepted_true.npy'.format(args.prefix_save_name,attack),index_array_accepted_true)
                    np.save('./attack_arrays/{}_{}_accepted_false.npy'.format(args.prefix_save_name,attack),index_array_accepted_false)
                    np.save('./attack_arrays_0%/{}_{}_accepted_true.npy'.format(args.prefix_save_name,attack),index_array_accepted_true_0)
                    np.save('./attack_arrays_0%/{}_{}_accepted_false.npy'.format(args.prefix_save_name,attack),index_array_accepted_false_0)
                    print("The number of images accepted and correctly classified 0% for {} is {}".format(attack,len(index_array_accepted_true_0)))
                    print("The number of images accepted and misclassified 0% for {} is{}".format(attack,len(index_array_accepted_false_0)))
                    print("The number of images accepted and correctly classified 10% for {} is {}".format(attack,len(index_array_accepted_true)))
                    print("The number of images accepted and misclassified 10% for {} is{}".format(attack,len(index_array_accepted_false)))
                    print("The number of images rejected 10% for {} is{}".format(attack,len(index_array_rejected)))

          

            ############  Results for the ensemble of attacks for 0% rejection ###############
            if args.attacks_to_run[0] == "MT" or args.attacks_to_run[0] == "Gama_MT":
              false_class_accepted = np.union1d(np.load('./attack_arrays_0%/{}_{}_accepted_false{}.npy'.format(args.prefix_save_name,args.attacks_to_run[0],0)),np.load('./attack_arrays_0%/{}_{}_accepted_false{}.npy'.format(args.prefix_save_name,args.attacks_to_run[0],0)))
            else:
              false_class_accepted = np.union1d(np.load('./attack_arrays_0%/{}_{}_accepted_false.npy'.format(args.prefix_save_name,args.attacks_to_run[0])),np.load('./attack_arrays_0%/{}_{}_accepted_false.npy'.format(args.prefix_save_name,args.attacks_to_run[0])))
            for attack in args.attacks_to_run:
                if attack =="Gama_MT":
                    for r in range(5):
                        false_class_accepted = np.union1d(false_class_accepted,np.load('./attack_arrays_0%/{}_{}_accepted_false{}.npy'.format(args.prefix_save_name,attack,r)))
                elif attack =="MT":
                    for r in range(10):
                        false_class_accepted = np.union1d(false_class_accepted,np.load('./attack_arrays_0%/{}_{}_accepted_false{}.npy'.format(args.prefix_save_name,attack,r)))
                else:
                    false_class_accepted = np.union1d(false_class_accepted,np.load('./attack_arrays_0%/{}_{}_accepted_false.npy'.format(args.prefix_save_name,attack)))
            print("The FW set for 0% rejection:",len(false_class_accepted))
            
            if args.attacks_to_run[0] == "MT" or args.attacks_to_run[0] == "Gama_MT":
              true_class_accepted = np.intersect1d(np.load('./attack_arrays_0%/{}_{}_accepted_true{}.npy'.format(args.prefix_save_name,args.attacks_to_run[0],0)),np.load('./attack_arrays_0%/{}_{}_accepted_true{}.npy'.format(args.prefix_save_name,args.attacks_to_run[0],0)))
            else:
              true_class_accepted = np.intersect1d(np.load('./attack_arrays_0%/{}_{}_accepted_true.npy'.format(args.prefix_save_name,args.attacks_to_run[0])),np.load('./attack_arrays_0%/{}_{}_accepted_true.npy'.format(args.prefix_save_name,args.attacks_to_run[0])))
            for attack in args.attacks_to_run:
                if attack =="Gama_MT":
                    for r in range(5):
                        true_class_accepted = np.intersect1d(true_class_accepted,np.load('./attack_arrays_0%/{}_{}_accepted_true{}.npy'.format(args.prefix_save_name,attack,r)))
                elif attack =="MT":
                    for r in range(10):
                        true_class_accepted = np.intersect1d(true_class_accepted,np.load('./attack_arrays_0%/{}_{}_accepted_true{}.npy'.format(args.prefix_save_name,attack,r)))
                else: 
                    true_class_accepted = np.intersect1d(true_class_accepted,np.load('./attack_arrays_0%/{}_{}_accepted_true.npy'.format(args.prefix_save_name,attack)))
            print("The FC set for 0% rejection:",len(true_class_accepted))
            print("Acc_0%",len(true_class_accepted)/(len(true_class_accepted)+len(false_class_accepted)))

            ###############  Results for the ensemble of attacks for a particular% of clean correctly classified images rejection ###############
            
            if args.attacks_to_run[0] == "MT" or args.attacks_to_run[0] == "Gama_MT":
              false_class_accepted = np.union1d(np.load('./attack_arrays/{}_{}_accepted_false{}.npy'.format(args.prefix_save_name,args.attacks_to_run[0],0)),np.load('./attack_arrays/{}_{}_accepted_false{}.npy'.format(args.prefix_save_name,args.attacks_to_run[0],0))) 
            else:
              false_class_accepted = np.union1d(np.load('./attack_arrays/{}_{}_accepted_false.npy'.format(args.prefix_save_name,args.attacks_to_run[0])),np.load('./attack_arrays/{}_{}_accepted_false.npy'.format(args.prefix_save_name,args.attacks_to_run[0])))
            for attack in args.attacks_to_run:
                if attack =="Gama_MT":
                    for r in range(5):
                        false_class_accepted = np.union1d(false_class_accepted,np.load('./attack_arrays/{}_{}_accepted_false{}.npy'.format(args.prefix_save_name,attack,r)))

                elif attack =="MT":
                    for r in range(10):
                        false_class_accepted = np.union1d(false_class_accepted,np.load('./attack_arrays/{}_{}_accepted_false{}.npy'.format(args.prefix_save_name,attack,r)))
                else:
                    false_class_accepted = np.union1d(false_class_accepted,np.load('./attack_arrays/{}_{}_accepted_false.npy'.format(args.prefix_save_name,attack)))
            print("The FW set for 10% rejection:",len(false_class_accepted))

            if args.attacks_to_run[0] == "MT" or args.attacks_to_run[0] == "Gama_MT":
              true_class_accepted = np.intersect1d(np.load('./attack_arrays/{}_{}_accepted_true{}.npy'.format(args.prefix_save_name,args.attacks_to_run[0],0)),np.load('./attack_arrays/{}_{}_accepted_true{}.npy'.format(args.prefix_save_name,args.attacks_to_run[0],0)))
            else:
              true_class_accepted = np.intersect1d(np.load('./attack_arrays/{}_{}_accepted_true.npy'.format(args.prefix_save_name,args.attacks_to_run[0])),np.load('./attack_arrays/{}_{}_accepted_true.npy'.format(args.prefix_save_name,args.attacks_to_run[0])))
            for attack in args.attacks_to_run:
                if attack =="Gama_MT":
                    for r in range(5):
                        true_class_accepted = np.intersect1d(true_class_accepted,np.load('./attack_arrays/{}_{}_accepted_true{}.npy'.format(args.prefix_save_name,attack,r)))
                elif attack =="MT":
                    for r in range(10):
                        true_class_accepted = np.intersect1d(true_class_accepted,np.load('./attack_arrays/{}_{}_accepted_true{}.npy'.format(args.prefix_save_name,attack,r)))
                else:
                    true_class_accepted = np.intersect1d(true_class_accepted,np.load('./attack_arrays/{}_{}_accepted_true.npy'.format(args.prefix_save_name,attack)))
            print("The FC set for 10% rejection:",len(true_class_accepted))
            print("Acc_10%",len(true_class_accepted)/(len(true_class_accepted)+len(false_class_accepted)))


            if args.attacks_to_run[0] == "MT" or args.attacks_to_run[0] == "Gama_MT":
              class_rejected = np.union1d(np.load('./attack_arrays/{}_{}_rejected{}.npy'.format(args.prefix_save_name,args.attacks_to_run[0],0)),np.load('./attack_arrays/{}_{}_rejected{}.npy'.format(args.prefix_save_name,args.attacks_to_run[0],0))) 
            else:
              class_rejected = np.union1d(np.load('./attack_arrays/{}_{}_rejected.npy'.format(args.prefix_save_name,args.attacks_to_run[0])),np.load('./attack_arrays/{}_{}_rejected.npy'.format(args.prefix_save_name,args.attacks_to_run[0])))
            for attack in args.attacks_to_run:
                if attack =="Gama_MT":
                    for r in range(5):
                        class_rejected = np.union1d(class_rejected,np.load('./attack_arrays/{}_{}_rejected{}.npy'.format(args.prefix_save_name,attack,r)))
                elif attack =="MT":
                    for r in range(10):
                        class_rejected = np.union1d(class_rejected,np.load('./attack_arrays/{}_{}_rejected{}.npy'.format(args.prefix_save_name,attack,r)))
                else:
                    class_rejected = np.union1d(class_rejected,np.load('./attack_arrays/{}_{}_rejected.npy'.format(args.prefix_save_name,attack)))
            print("Rejection_10%",len(class_rejected))