import math
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

parser.add_argument('--source_model', type=str, default='resnet50', choices=['resnet50', 'inception-v3', 'densenet121', 'vgg16bn'])

parser.add_argument('--batch_size', type=int, default=50)
parser.add_argument('--max_iterations', type=int, default=400)

parser.add_argument('--loss_function', type=str, default='CE', choices=['CE','MaxLogit'])

parser.add_argument('--targeted', action='store_true')

parser.add_argument('--m1', type=int, default=1, help='number of randomly sampled images')
parser.add_argument('--m2', type=int, default=1, help='num of copies')
parser.add_argument('--strength', type=float, default=0)

parser.add_argument('--adv_perturbation', action='store_true')

parser.add_argument('--adv_loss_function', type=str, default='CE', choices=['CE', 'MaxLogit'])

parser.add_argument('--adv_epsilon', type=eval, default=16/255)
parser.add_argument('--adv_steps', type=int, default=8)

parser.add_argument('--transpoint', type=int, default=0)

parser.add_argument('--seed', type=int, default=0)


parser.add_argument('--MI', action='store_true')
parser.add_argument('--DI', action='store_true')
parser.add_argument('--TI', action='store_true')
parser.add_argument('--SI', action='store_true')
parser.add_argument('--random_start', action='store_true')


parser.add_argument('--save', action='store_true')

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


exp_name = arg.source_model + '_' + arg.loss_function + '_'

if arg.targeted:
    exp_name += 'T_'
if arg.MI:
    exp_name += 'MI_'
if arg.DI:
    exp_name += 'DI_'
if arg.TI:
    exp_name += 'TI_'
if arg.SI:
    exp_name += 'SI_'
if arg.m1 != 1:
    exp_name += f'm1_{arg.m1}_'
if arg.m2 != 1:
    exp_name += f'm2_{arg.m2}_'
if arg.strength != 0:
    exp_name += 'Admix_'


exp_name += str(arg.transpoint)


if arg.targeted:
    exp_name += '_target'


# for targeted attack, we need to conduct the untargeted attack during the inner loop.
# for untargeted attack, we need to conduct the targeted attack (the true label) during the inner loop. 
if not arg.targeted:
    arg.adv_targeted = 1
else:
    arg.adv_targeted = 0


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
    if c <= 0.7:
        X_out = F.pad(F.interpolate(X_in, size=(rnd, rnd)), (pad_left, pad_right, pad_top, pad_bottom), mode='constant', value=0)
        return X_out
    else:
        return X_in


def pgd(model, data, labels, targeted, epsilon, k, a, random_start=True):
    
    data_max = data + epsilon
    data_min = data - epsilon
    data_max.clamp_(0, 1)
    data_min.clamp_(0, 1)

    data = data.clone().detach().to(device)
    labels = labels.clone().detach().to(device)

    perturbed_data = data.clone().detach()

    if random_start:
        # Starting at a uniformly random point
        perturbed_data = perturbed_data + torch.empty_like(perturbed_data).uniform_(-epsilon, epsilon)
        perturbed_data = torch.clamp(perturbed_data, min=0, max=1).detach()

    for _ in range(k):
        perturbed_data.requires_grad = True
        outputs = model(norm(perturbed_data))
        if arg.adv_loss_function == 'CE':
            loss = nn.CrossEntropyLoss(reduction='sum')
            if targeted:
                cost = loss(outputs, labels)
            else:
                cost = -1 * loss(outputs, labels)


        elif arg.adv_loss_function == 'MaxLogit':
            if targeted:
                real = outputs.gather(1, labels.unsqueeze(1)).squeeze(1)
                logit_dists = -1 * real
                cost = logit_dists.sum()
            else:
                real = outputs.gather(1, labels.unsqueeze(1)).squeeze(1)
                cost = real.sum()

        # Update adversarial images
        cost.backward()

        gradient = perturbed_data.grad.clone().to(device)
        perturbed_data.grad.zero_()

        with torch.no_grad():
            perturbed_data.data -= a * torch.sign(gradient)
            perturbed_data.data = torch.max(torch.min(perturbed_data.data, data_max), data_min)
    return perturbed_data.detach()


model_1 = models.inception_v3(pretrained=True, transform_input=True).eval()
model_2 = models.resnet50(pretrained=True).eval()
model_3 = models.densenet121(pretrained=True).eval()
model_4 = models.vgg16_bn(pretrained=True).eval()


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

if arg.source_model == 'inception-v3':
    model_source = model_1
elif arg.source_model == 'resnet50':
    model_source = model_2
elif arg.source_model == 'densenet121':
    model_source = model_3
elif arg.source_model == 'vgg16bn':
    model_source = model_4

logging("setting up the source and target models")

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

logging("loaded the images".format())
n = tdist.Normal(0.0, 15/255)

#-------------------------------------#
X_adv_10 = torch.zeros(len(image_id_list), 3, img_size, img_size)
X_adv_50 = torch.zeros(len(image_id_list), 3, img_size, img_size)
X_adv_100 = torch.zeros(len(image_id_list), 3, img_size, img_size)
X_adv_200 = torch.zeros(len(image_id_list), 3, img_size, img_size)
X_adv_300 = torch.zeros(len(image_id_list), 3, img_size, img_size)
X_adv_400 = torch.zeros(len(image_id_list), 3, img_size, img_size)

fixing_point = 0

adv_activate = 0

pos = np.zeros((4, arg.max_iterations // 10))


for k in range(0, num_batches):
    batch_size_cur = min(arg.batch_size, len(image_id_list) - k * arg.batch_size)
    X_ori = torch.zeros(batch_size_cur, 3, img_size, img_size).to(device)
    delta = torch.zeros_like(X_ori, requires_grad=True).to(device)
    for i in range(batch_size_cur):
        X_ori[i] = trn(Image.open(input_path + image_id_list[k * arg.batch_size + i] + '.png'))
    labels = torch.tensor(label_ori_list[k * arg.batch_size:k * arg.batch_size + batch_size_cur]).to(device)
    target_labels = torch.tensor(label_tar_list[k * arg.batch_size:k * arg.batch_size + batch_size_cur]).to(device)
    grad_pre = 0
    prev = float('inf')

    if arg.random_start:
        # Starting at a uniformly random point
        delta.requires_grad_(False)
        delta = delta + torch.empty_like(X_ori).uniform_(-epsilon/255, epsilon/255)
        delta = torch.clamp(X_ori+delta, min=0, max=1) - X_ori
        delta.requires_grad_(True)

    logging(50*"#")
    logging("starting :{} batch".format(k+1))
    

    for t in range(arg.max_iterations):
        if t < arg.transpoint:
            adv_activate = 0
        else:
            if arg.adv_perturbation:
                adv_activate = 1
            else:
                adv_activate = 0
        grad_list = []

        for q in range(arg.m1):
            delta.requires_grad_(False)

            if arg.strength == 0:
                X_addin = torch.zeros_like(X_ori).to(device)
            else:
                X_addin = torch.zeros_like(X_ori).to(device)
                random_labels = torch.zeros(batch_size_cur).to(device)
                stop = False
                while stop == False:
                    random_indices = np.random.randint(0, 1000, batch_size_cur)
                    for i in range(batch_size_cur):
                        X_addin[i] = trn(Image.open(input_path + image_id_list[random_indices[i]] + '.png'))
                        random_labels[i] = label_ori_list[random_indices[i]]
                    if torch.sum(random_labels==labels).item() == 0:
                        stop = True
                X_addin = arg.strength * X_addin
                X_addin = torch.clamp(X_ori+delta+X_addin, min=0, max=1) - (X_ori+delta)
            
            if arg.SI:

                if adv_activate:
                    top_values_1, top_indices_1 = model_source(norm(X_ori+delta+X_addin)).topk(arg.m1+1, dim=1, largest=True, sorted=True)
                    
                    if arg.adv_targeted:
                        label_pred = labels
                    else:
                        label_pred = target_labels

                    X_advaug = pgd(model_source, X_ori+delta+X_addin, label_pred, arg.adv_targeted, arg.adv_epsilon, arg.adv_steps, arg.adv_alpha)
                    X_aug = X_advaug - (X_ori+delta+X_addin)

                else:
                    X_aug = torch.zeros_like(X_ori).to(device)

            delta.requires_grad_(True)

            for j in range(arg.m2):

                if not arg.SI:
                    delta.requires_grad_(False)

                    if adv_activate:
                        top_values_2, top_indices_2 = model_source(norm(X_ori+delta+X_addin)).topk(arg.m2+1, dim=1, largest=True, sorted=True)
                        
                        if arg.adv_targeted:
                            label_pred = labels
                        else:
                            label_pred = target_labels
                        
                        X_advaug = pgd(model_source, X_ori+delta+X_addin, label_pred, arg.adv_targeted, arg.adv_epsilon, arg.adv_steps, arg.adv_alpha)
                        X_aug = X_advaug - (X_ori+delta+X_addin)

                    else:
                        X_aug = torch.zeros_like(X_ori).to(device)
                    delta.requires_grad_(True)

                if arg.DI:  # DI
                    if arg.SI:
                        logits = model_source(norm(DI((X_ori + delta + X_addin + X_aug )/2**j)))
                    else:
                        logits = model_source(norm(DI(X_ori + delta + X_addin + X_aug )))
                else:
                    if arg.SI:
                        logits = model_source(norm((X_ori + delta + X_addin + X_aug )/2**j))
                    else:
                        logits = model_source(norm(X_ori + delta + X_addin + X_aug ))

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

        for j in range(arg.m1 * arg.m2):
            grad_c += grad_list[j]
        grad_c = grad_c / (arg.m1 * arg.m2)

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



        if t == (1-1):
            X_adv_10[fixing_point:fixing_point+batch_size_cur] = (X_ori + delta).clone().detach().cpu()
        if t == (50-1):
            X_adv_50[fixing_point:fixing_point+batch_size_cur] = (X_ori + delta).clone().detach().cpu()
        if t == (100-1):
            X_adv_100[fixing_point:fixing_point+batch_size_cur] = (X_ori + delta).clone().detach().cpu()
        if t == (200-1):
            X_adv_200[fixing_point:fixing_point+batch_size_cur] = (X_ori + delta).clone().detach().cpu()
        if t == (300-1):
            X_adv_300[fixing_point:fixing_point+batch_size_cur] = (X_ori + delta).clone().detach().cpu()
        if t == (400-1):
            X_adv_400[fixing_point:fixing_point+batch_size_cur] = (X_ori + delta).clone().detach().cpu()
    
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
    X_adv_400 = X_adv_400.detach().cpu()

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

