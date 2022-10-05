from copy import deepcopy
import math
# from tkinter import X
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.autograd as autograd
import torchvision
import torchvision.datasets as td
import torch.distributions as tdist
import argparse
from torchvision import models, transforms
from PIL import Image
import csv
import numpy as np
import os
import scipy.stats as st

## hyperparameter
parser = argparse.ArgumentParser()

parser.add_argument('--batch_size', type=int, default=40)
parser.add_argument('--max_iterations', type=int, default=400)
parser.add_argument('--seed', type=int, default=23232)

parser.add_argument('--loss_function', type=str, default='CE', choices=['CE','MaxLogit'])
parser.add_argument('--num_data_augmentation', type=int, default=1)
parser.add_argument('--transformation_rate', type=float, default=0.7)

parser.add_argument('--targeted', action='store_true')


parser.add_argument('--source_model', type=str, default='resnet_50', choices=['inception_v3','resnet_50','densenet_121','vgg16_bn'])
parser.add_argument('--adv_attack_method', type=str, default='PGD', choices=['PGD'])
parser.add_argument('--adv_epsilon', type=eval, default=16/255)
parser.add_argument('--adv_steps', type=int, default=8)

parser.add_argument('--transpoint', type=int, default=0)



parser.add_argument('--save', type=int, default=0)

parser.add_argument('--MI', action='store_true')
parser.add_argument('--DI', action='store_true')
parser.add_argument('--TI', action='store_true')
parser.add_argument('--SI', action='store_true')
parser.add_argument('--SI_number', type=int, default=1)

parser.add_argument('--device', type=int, default=0)

arg = parser.parse_args()


os.environ["CUDA_VISIBLE_DEVICES"] = str(arg.device)

arg.adv_alpha = arg.adv_epsilon / arg.adv_steps


def makedir(path):
    folder = os.path.exists(path)
    if not folder:
        os.makedirs(path)
        print('----------- new folder ------------')
        print('------------ ok ----------------')
    
    else:
        print('----------- There is this folder! -----------')



exp_name = arg.source_model+'_'+arg.loss_function+'_'+ 'iter_'+str(arg.max_iterations)+'_'


if arg.MI:
    exp_name += 'MI_'
if arg.DI:
    exp_name += 'DI_'
if arg.TI:
    exp_name += 'TI_'
if arg.SI:
    exp_name += 'SI_'
    exp_name = exp_name + str(arg.SI_number) + '_'

exp_name += 'num_aug_'
exp_name += str(arg.num_data_augmentation) + '_'

    
exp_name += str(arg.transpoint)

if arg.targeted:
    exp_name += '_target'
    


if arg.save:

    arg.file_path = "/targeted_attack/adv_example/"+exp_name

    makedir(arg.file_path)


def logging(s, print_=True, log_=True):

    if print_:
        print(s)

    if log_:
        with open(os.path.join(arg.file_path, 'log.txt'), 'a+') as f_log:
            f_log.write(s + '\n')



logging(exp_name.format())

logging('Hyper-parameters: {}\n'.format(arg.__dict__))


##load image metadata (Image_ID, true label, and target label)
def load_ground_truth(csv_filename):
    image_id_list = []
    label_ori_list = []
    label_tar_list = []

    with open(csv_filename) as csvfile:
        reader = csv.DictReader(csvfile, delimiter=',')
        for row in reader:
            image_id_list.append(row['ImageId'])
            label_ori_list.append(int(row['TrueLabel']) - 1)
            label_tar_list.append(int(row['TargetClass']) - 1)

    return image_id_list, label_ori_list, label_tar_list

## simple Module to normalize an image
class Normalize(nn.Module):
    def __init__(self, mean, std):
        super(Normalize, self).__init__()
        self.mean = torch.Tensor(mean)
        self.std = torch.Tensor(std)

    def forward(self, x):
        return (x - self.mean.type_as(x)[None, :, None, None]) / self.std.type_as(x)[None, :, None, None]

##define TI
def gkern(kernlen=15, nsig=3):
    x = np.linspace(-nsig, nsig, kernlen)
    kern1d = st.norm.pdf(x)
    kernel_raw = np.outer(kern1d, kern1d)
    kernel = kernel_raw / kernel_raw.sum()
    return kernel
channels = 3
kernel_size = 5
kernel = gkern(kernel_size, 3).astype(np.float32)
gaussian_kernel = np.stack([kernel, kernel, kernel])
gaussian_kernel = np.expand_dims(gaussian_kernel, 1)
gaussian_kernel = torch.from_numpy(gaussian_kernel).cuda()

##define DI
def DI(X_in):
    rnd = np.random.randint(299, 330, size=1)[0]
    h_rem = 330 - rnd
    w_rem = 330 - rnd
    pad_top = np.random.randint(0, h_rem, size=1)[0]
    pad_bottom = h_rem - pad_top
    pad_left = np.random.randint(0, w_rem, size=1)[0]
    pad_right = w_rem - pad_left

    c = np.random.rand(1)
    if c <= arg.transformation_rate:
        X_out = F.pad(F.interpolate(X_in, size=(rnd, rnd)), (pad_left, pad_right, pad_top, pad_bottom), mode='constant', value=0)
        return X_out
    else:
        return X_in




def pgd(model, data, labels, random_start=True):
    
    epsilon = arg.adv_epsilon
    k = arg.adv_steps
    a = arg.adv_alpha
    data_max = data + epsilon
    data_min = data - epsilon
    d_min=0
    d_max=1
    data_max.clamp_(d_min, d_max)
    data_min.clamp_(d_min, d_max)

    data = data.clone().detach().to(device)
    labels = labels.clone().detach().to(device)

    perturbed_data = data.clone().detach()

    if random_start:
        # Starting at a uniformly random point
        perturbed_data = perturbed_data + torch.empty_like(perturbed_data).uniform_(-1*epsilon, epsilon)
        perturbed_data = torch.clamp(perturbed_data, min=0, max=1).detach()

    for _ in range(k):
        perturbed_data.requires_grad = True
        outputs = model(norm(perturbed_data))

        loss = nn.CrossEntropyLoss(reduction='sum')
        cost = -1 * loss(outputs, labels)
        # Update adversarial images
        cost.backward()

        gradient = perturbed_data.grad.clone().to(device)
        perturbed_data.grad.zero_()
        with torch.no_grad():
            perturbed_data.data -= a * torch.sign(gradient)
            perturbed_data.data = torch.max(torch.min(perturbed_data, data_max), data_min)
    return perturbed_data




model_1 = models.inception_v3(pretrained=True, transform_input=True).eval()
model_2 = models.resnet50(pretrained=True).eval()
model_3 = models.densenet121(pretrained=True).eval()
model_4 = models.vgg16_bn(pretrained=True).eval()


model_1_n = "inception_v3"
model_2_n = "resnet_50"
model_3_n = "densenet_121"
model_4_n = "vgg16_bn"


for param in model_1.parameters():
    param.requires_grad = False
for param in model_2.parameters():
    param.requires_grad = False
for param in model_3.parameters():
    param.requires_grad = False
for param in model_4.parameters():
    param.requires_grad = False


device = 'cuda' if torch.cuda.is_available() else 'cpu'
logging(f'device: {device}')
model_1.to(device)
model_2.to(device)
model_3.to(device)
model_4.to(device)

torch.manual_seed(arg.seed)
torch.backends.cudnn.deterministic = True
np.random.seed(arg.seed)


# values are standard normalization for ImageNet images,
# from https://github.com/pytorch/examples/blob/master/imagenet/main.py
norm = Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
trn = transforms.Compose([transforms.ToTensor(), ])
image_id_list, label_ori_list, label_tar_list = load_ground_truth('/targeted_attack/dataset/images.csv')

img_size = 299
input_path = '/targeted_attack/dataset/images/'
lr = 2 / 255  # step size
epsilon = 16  # L_inf norm bound
num_batches = np.int(np.ceil(len(image_id_list) / arg.batch_size))
logging("loaded the images")

if arg.source_model == "resnet_50":
    model_source  = model_2
    model_source_n = model_2_n
elif arg.source_model == "inception_v3":
    model_source  = model_1
    model_source_n = model_1_n
elif arg.source_model == "densenet_121":
    model_source  = model_3
    model_source_n = model_3_n
elif arg.source_model == "vgg16_bn":
    model_source  = model_4
    model_source_n = model_4_n
logging("setting up the source and target models")

#-------------------------------------#
X_adv_10 = torch.zeros(len(image_id_list),3,img_size,img_size).to(device)
X_adv_50 = torch.zeros(len(image_id_list),3,img_size,img_size).to(device)
X_adv_100 = torch.zeros(len(image_id_list),3,img_size,img_size).to(device)
X_adv_200 = torch.zeros(len(image_id_list),3,img_size,img_size).to(device)
X_adv_300 = torch.zeros(len(image_id_list),3,img_size,img_size).to(device)
X_adv_final = torch.zeros(len(image_id_list),3,img_size,img_size).to(device)

fixing_point = 0

adv_activate = 0
pos = np.zeros((4, arg.max_iterations // 10))


attack_x_adv_0 = 0
attack_x_adv_10 = 0
attack_x_adv_20 = 0
attack_x_adv_50 = 0
attack_x_adv_100 = 0
attack_x_adv_200 = 0

gau_x_adv_0 = 0
gau_x_adv_10 = 0
gau_x_adv_20 = 0
gau_x_adv_50 = 0
gau_x_adv_100 = 0
gau_x_adv_200 = 0

total_number = 0

for k in range(0, num_batches):

    batch_size_cur = min(arg.batch_size, len(image_id_list) - k * arg.batch_size)
    total_number += batch_size_cur
    X_ori = torch.zeros(batch_size_cur, 3, img_size, img_size).to(device)
    delta = torch.zeros_like(X_ori, requires_grad=True).to(device)
    for i in range(batch_size_cur):
        X_ori[i] = trn(Image.open(input_path + image_id_list[k * arg.batch_size + i] + '.png'))
    labels = torch.tensor(label_ori_list[k * arg.batch_size:k * arg.batch_size + batch_size_cur]).to(device)
    target_labels = torch.tensor(label_tar_list[k * arg.batch_size:k * arg.batch_size + batch_size_cur]).to(device)
    grad_pre = 0
    prev = float('inf')

    logging(50*"#")
    logging("starting :{} batch".format(k+1))

    for t in range(arg.max_iterations):


        if t < arg.transpoint:
            adv_activate = 0
            num_augmentation = 1
        else:
            adv_activate = 1
            num_augmentation = arg.num_data_augmentation
        
        
        grad_list = []
        
        for ii in range(num_augmentation):

            if adv_activate:
                #-----------------------------------#
                # do untargeted attacks for adversarial data augmentations
                delta.requires_grad_(False)
                
                if arg.adv_targeted:
                    label_pred = labels
                else:
                    label_pred = target_labels
                
                if arg.adv_attack_method == 'PGD':
                    X_advaug = pgd(model_source, X_ori+delta, label_pred)
                
                X_aug = X_advaug - (X_ori+delta)
                delta.requires_grad_(True)

            else:
                X_aug = 0

            for j in range(arg.SI_number):
                
                if arg.DI:  # DI
                    if arg.SI:
                        logits = model_source(norm(DI((X_ori + X_aug + delta)/2**j)))
                    else:
                        logits = model_source(norm(DI(X_ori + X_aug + delta)))
                else:
                    if arg.SI:
                        logits = model_source(norm((X_ori + delta + X_aug)/2**j))
                    else:
                        logits = model_source(norm(X_ori + delta + X_aug))

                if arg.loss_function == 'CE':
                    loss_func = nn.CrossEntropyLoss(reduction='sum')
                    if arg.targeted:
                        loss = loss_func(logits, target_labels)
                    else:
                        loss = -1 * loss_func(logits, labels)
                elif arg.loss_function == 'MaxLogit':
                    if arg.targeted:
                        real = logits.gather(1,target_labels.unsqueeze(1)).squeeze(1)
                        loss = -1 * real.sum()
                    else:
                        real = logits.gather(1,labels.unsqueeze(1)).squeeze(1)
                        loss = real.sum()
                loss.backward()
                grad_cc = delta.grad.clone().to(device)
                if arg.TI:  # TI
                    grad_cc = F.conv2d(grad_cc, gaussian_kernel, bias=None, stride=1, padding=(2, 2), groups=3)
                grad_list.append(grad_cc)
                delta.grad.zero_()

        grad_c = 0
        for j in range(len(grad_list)):
            grad_c += grad_list[j]
        grad_c = grad_c / len(grad_list)

        if arg.MI:  # MI
            grad_c = grad_c / torch.mean(torch.abs(grad_c), (1, 2, 3), keepdim=True) + 1 * grad_pre
            
        grad_pre = grad_c
        delta.data = delta.data - lr * torch.sign(grad_c)
        delta.data = delta.data.clamp(-epsilon / 255, epsilon / 255)
        delta.data = ((X_ori + delta.data).clamp(0, 1)) - X_ori

        if t % 10 == 9:
            if arg.targeted:
                pos[0, t // 10] = pos[0, t // 10] + sum(torch.argmax(model_1(norm(X_ori + delta)), dim=1) == target_labels).cpu().numpy()
                pos[1, t // 10] = pos[1, t // 10] + sum(torch.argmax(model_2(norm(X_ori + delta)), dim=1) == target_labels).cpu().numpy()
                pos[2, t // 10] = pos[2, t // 10] + sum(torch.argmax(model_3(norm(X_ori + delta)), dim=1) == target_labels).cpu().numpy()
                pos[3, t // 10] = pos[3, t // 10] + sum(torch.argmax(model_4(norm(X_ori + delta)), dim=1) == target_labels).cpu().numpy()
            else:
                pos[0, t // 10] = pos[0, t // 10] + sum(torch.argmax(model_1(norm(X_ori + delta)), dim=1) != labels).cpu().numpy()
                pos[1, t // 10] = pos[1, t // 10] + sum(torch.argmax(model_2(norm(X_ori + delta)), dim=1) != labels).cpu().numpy()
                pos[2, t // 10] = pos[2, t // 10] + sum(torch.argmax(model_3(norm(X_ori + delta)), dim=1) != labels).cpu().numpy()
                pos[3, t // 10] = pos[3, t // 10] + sum(torch.argmax(model_4(norm(X_ori + delta)), dim=1) != labels).cpu().numpy()
            
            logging(str(pos))
            logging(30*"#")

        # gau_alpha = 0.05
        # save adv examples in 10, 50, 100, 200, 300, final iters 
        if t == (1-1):
            X_adv_10[fixing_point: fixing_point+batch_size_cur] = (X_ori + delta).detach()

        if t == (20-1):
            # x_test = (X_ori + delta).detach()
            X_adv_10[fixing_point: fixing_point+batch_size_cur] = (X_ori + delta).detach()

        if t == (50-1):
            X_adv_50[fixing_point: fixing_point+batch_size_cur] = (X_ori + delta).detach()

        if t == (100-1):
            X_adv_100[fixing_point: fixing_point+batch_size_cur] = (X_ori + delta).detach()
        if t == (200-1):
            X_adv_200[fixing_point: fixing_point+batch_size_cur] = (X_ori + delta).detach()
        if t == (300-1):
            X_adv_300[fixing_point: fixing_point+batch_size_cur] = (X_ori + delta).detach()
        if t == (arg.max_iterations-1):
            X_adv_final[fixing_point: fixing_point+batch_size_cur] = (X_ori + delta).detach()

    fixing_point += batch_size_cur
    logging(50*"#")

torch.cuda.empty_cache()


logging(arg.file_path.format())

logging('Hyper-parameters: {}\n'.format(arg.__dict__))


logging("final result")
logging('Source model : Ensemble --> Target model: Inception-v3 | ResNet50 | DenseNet121 | VGG16bn')
logging(str(pos))

logging("results for 10 iters:")
logging(str(pos[:, 0]))

logging("results for 100 iters:")
logging(str(pos[:, 9]))

logging("results for 200 iters:")
logging(str(pos[:, 19]))

logging("results for 300 iters:")
logging(str(pos[:, 29]))

logging("results for 400 iters:")
logging(str(pos[:, 39]))



if arg.save:
    np.save(arg.file_path+'/'+'results'+'.npy', pos)


    # X_adv_10 = X_adv_10.detach().cpu()
    # X_adv_50 = X_adv_50.detach().cpu()
    # X_adv_100 = X_adv_100.detach().cpu()
    # X_adv_200 = X_adv_200.detach().cpu()
    # X_adv_300 = X_adv_300.detach().cpu()
    X_adv_400 = X_adv_final.detach().cpu()

    logging("saving the adversarial examples")

    # torch.save(X_adv_10, file_path+'/'+'iter_10'+'.pt')
    # np.save(file_path+'/'+'iter_10'+'.npy', X_adv_10.numpy())

    # # torch.save(X_adv_50, file_path+'/'+'iter_50'+'.pt')
    # np.save(file_path+'/'+'iter_50'+'.npy', X_adv_50.numpy())

    # torch.save(X_adv_100, file_path+'/'+'iter_100'+'.pt')
    # np.save(file_path+'/'+'iter_100'+'.npy', X_adv_100.numpy())

    # # torch.save(X_adv_200, file_path+'/'+'iter_200'+'.pt')
    # np.save(file_path+'/'+'iter_200'+'.npy', X_adv_200.numpy())

    # torch.save(X_adv_300, file_path+'/'+'iter_300'+'.pt')
    # np.save(file_path+'/'+'iter_300'+'.npy', X_adv_300.numpy())

    # torch.save(X_adv_final, file_path+'/'+'iter_final'+'.pt')
    np.save(arg.file_path+'/'+'iter_400'+'.npy', X_adv_400.numpy())

    logging("finishing saving the adversarial examples")

logging("finishing the attack experiment")
logging(50*"#")
logging(50*"#")
logging(50*"#")









