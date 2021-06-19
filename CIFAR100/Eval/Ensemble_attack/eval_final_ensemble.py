import os
import argparse   
import torch
import torch.nn as nn
import torchvision.datasets as datasets
import torch.utils.data as data
import torchvision.transforms as transforms
import numpy as np
import sys
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
parser.add_argument('--model', type=str, default='../../../FLSS_cifar100.pt')
parser.add_argument('--n_ex', type=int, default=10000)
parser.add_argument('--restarts', type=int, default=1)
parser.add_argument('--individual', action='store_true')
parser.add_argument('--cheap', action='store_true')
parser.add_argument('--save_dir', type=str, default='./results')
parser.add_argument('--batch_size', type=int, default=1000)
parser.add_argument('--plus', action='store_true')
parser.add_argument('--attacks_to_run', type=str, nargs='*', default=["clean","apgd_ce","apgd_dlr","fab","square","MM","GAMA_pgd","pgd","Gama_MT","MT"  ])
parser.add_argument('--balanced_test_size', type=int, default=1000)
parser.add_argument('--threshold', type=float, default=56)
parser.add_argument('--take_threshold_from_file', type=bool, default=True)
parser.add_argument('--type_of_thresholder', type=str, default="original")
parser.add_argument('--full_test_eval', type=bool, default=False)
parser.add_argument('--no_state_dict', type=bool, default=True)
parser.add_argument('--prefix_save_name', type=str, default="index_array")
args = parser.parse_args()
if __name__ == '__main__':
            ############  Results for the ensemble of attacks for 0% rejection ###############
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