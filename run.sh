cd ./CIFAR10/Train
python train_cifar10.py | tee -a train_cifar10.txt

cd ../Eval/Threshold_finder
python eval_test.py --model ../../Train/model-cifar-ResNet/results/best_model-res18FC.pt | tee -a threshold_cifar10.txt

cd ../Ensemble_attack
python eval.py --model ../../Train/model-cifar-ResNet/results/best_model-res18FC.pt  --prefix_save_name cifar10 | tee -a eval_cifar10.txt

cd ./CIFAR100/Train
python train_cifar100_cyclic_lr.py | tee -a train_cifar100.txt

cd ../Eval/Threshold_finder
python eval_test.py --model ../../Train/model-cifar-ResNet/results/best_model-res18FC.pt | tee -a threshold_cifar100.txt

cd ../Ensemble_attack
python eval.py --model ../../Train/model-cifar-ResNet/results/best_model-res18FC.pt  --prefix_save_name cifar10 | tee -a eval_cifar100.txt
