# Boosting the Transferability of Adversarial Attacks with Reverse Adversarial Perturbation (NeurIPS 2022)
 

**PyTorch implementation for the state-of-art transfer attack: Reverse Adversarial Perturbation (RAP).**

*Boosting the Transferability of Adversarial Attacks with Reverse Adversarial Perturbation*

Zeyu Qin*, Yanbo Fan*, Yi Liu, Li Shen, Yong Zhang, Jue Wang, Baoyuan Wu

In NeurIPS 2022.

----

### Codes:
 - rap_attack_new.py: simple version
 - rap_attack.py: full version


### The examples:

- targeted attack with DI and logit loss from ResNet-50

    ```

    python /targeted_attack/rap_attack_new.py --num_data_augmentation 1  --targeted  --transpoint 400 --seed 9018 --source_model resnet_50 --loss_function MaxLogit --DI --max_iterations 300
    ```


- RAP targeted attack with DI and logit loss from ResNet-50

    ```
    python /targeted_attack/rap_attack_new.py --num_data_augmentation 1  --targeted  --transpoint 0 --seed 9018 --source_model resnet_50 --loss_function MaxLogit --DI --max_iterations 300
    ```


- RAP-LS targeted attack with DI and logit loss from ResNet-50

    ```
    python /targeted_attack/rap_attack_new.py --num_data_augmentation 1  --targeted  --transpoint 100 --seed 9018 --source_model resnet_50 --loss_function MaxLogit --DI --max_iterations 300
    ```

### The parameters of config:

    ```
    - targeted attack or not : --targeted or None
    - source model: -- source_model (resnet_50, densenet, inception, vgg16)
    - random seed: --seed 1234
    - interation number of outer minimization: --max_iterations 
    - MI or not: --MI or None
    - DI or not: --DI or None
    - TI or not: --TI or None
    - SI or not: (--SI and --m2 5) or None 
    - Admix or not: 
      (--m1 3 an --m2 5) or None
      --strength 0.2
    - transpoint:
      --transpoint 400: baseline method
      --transpoint 0: baseline+RAP
      --transpoint 100: baseline+RAP-LS
    - loss function: --loss_function: CE or MaxLogit for outer minimization
    - epsilon of attacks: --adv_epsilon: 16/255, the perturbation budget for - inner maximization
      --adv_steps: 8, the step for inner maximization
    ```

#### This code is based on [source code from NeurIPS 2021 paper](https://github.com/ZhengyuZhao/Targeted-Tansfer) , *"On Success and Simplicity: A Second Look at Transferable Targeted Attacks"*. The used dataset is also contained in this repository. Please consider leaving a :star: on their repository.
