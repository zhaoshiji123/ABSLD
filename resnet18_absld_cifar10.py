import os
import argparse
import torch
from absld_loss import *
from cifar10_models import *
import torchvision
from torchvision import datasets, transforms
import time
from torch.utils.data.sampler import SubsetRandomSampler

# we fix the random seed to 0, this method can keep the results consistent in the same conputer. 
torch.manual_seed(0)
torch.cuda.manual_seed_all(0)
torch.backends.cudnn.deterministic = True

os.environ["CUDA_VISIBLE_DEVICES"] = "0"



epochs = 300
batch_size = 128
epsilon = 8/255.0

transform_train = transforms.Compose([
    transforms.RandomCrop(32, padding=4),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
])
transform_test = transforms.Compose([
    transforms.ToTensor(),
])

trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=False, transform=transform_train)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, shuffle=True, num_workers=2)

testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=False, transform=transform_test)

ori_label = torch.tensor([y for (_, y) in testset])
n = 10000
valid_index = []

for i in range(10):
    valid_index_i = (ori_label == i).nonzero()[:n]
    valid_index.append(valid_index_i)
valid_index = torch.cat(valid_index, dim=0).flatten()

valid_sampler = SubsetRandomSampler(valid_index)

testloader = torch.utils.data.DataLoader(testset, batch_size=100, shuffle=False,sampler=valid_index, num_workers=2)

student = resnet18()
student = student.cuda()
student.train()
optimizer = optim.SGD(student.parameters(), lr=0.1, momentum=0.9, weight_decay=2e-4)



def kl_loss(a,b):
    return -a*b + torch.log(b+1e-5)*b

teacher = wideresnet()
teacher.load_state_dict(torch.load('./models/model_cifar_wrn.pt'))
teacher = teacher.cuda()
teacher.eval()

class_labels_temp_adv = torch.ones(10, dtype=torch.float).cuda()
class_labels_temp_nat = torch.ones(10, dtype=torch.float).cuda()

class_labels_temp_rate = 0.1

temp_max = 5
temp_min = 0.5

pgd_best_accuracy = 0
pgd_fair_best_accuracy = 0

ce_loss = torch.nn.CrossEntropyLoss(reduce = False).cuda()

start_epochs = 0



for epoch in range(start_epochs + 1,epochs+1):
    print('the {}th epoch '.format(epoch))
    class_labels_total_loss_adv = torch.zeros(10, dtype=torch.float).cuda()
    class_labels_total_loss_nat = torch.zeros(10, dtype=torch.float).cuda()


    for step,(train_batch_data,train_batch_labels) in enumerate(trainloader):
        student.train()
        train_batch_data = train_batch_data.float().cuda()
        train_batch_labels = train_batch_labels.cuda()
        optimizer.zero_grad()
        with torch.no_grad():
            teacher_logits = teacher(train_batch_data).detach()
            teacher_logits_new_adv = teacher_logits.detach()
            teacher_logits_new_nat = teacher_logits.detach()
            for sample_index in range(train_batch_labels.shape[0]):
                teacher_logits_new_adv[sample_index] = teacher_logits_new_adv[sample_index] / class_labels_temp_adv[train_batch_labels[sample_index]]
                teacher_logits_new_nat[sample_index] = teacher_logits_new_nat[sample_index] / class_labels_temp_nat[train_batch_labels[sample_index]]



        adv_logits = absld_inner_loss(student,teacher_logits_new_adv,train_batch_data,train_batch_labels,optimizer,step_size=2/255.0,epsilon=epsilon,perturb_steps=10)
        student.train()
        nat_logits = student(train_batch_data)
        adv_logit_1 = adv_logits.detach()
        nat_logit_1 = nat_logits.detach()
        # record
        with torch.no_grad():

 
            kl_Loss1_record = ce_loss(adv_logit_1.detach(), train_batch_labels.detach())
            kl_Loss2_record = ce_loss(nat_logit_1.detach(), train_batch_labels.detach())


            for sample_index in range(train_batch_labels.shape[0]):
                class_labels_total_loss_adv[train_batch_labels[sample_index]] =class_labels_total_loss_adv[train_batch_labels[sample_index]] + kl_Loss1_record[sample_index]
                class_labels_total_loss_nat[train_batch_labels[sample_index]] =class_labels_total_loss_nat[train_batch_labels[sample_index]] + kl_Loss2_record[sample_index]

        kl_Loss1 = kl_loss(torch.log(F.softmax(adv_logits,dim=1)),F.softmax(teacher_logits_new_adv.detach(),dim=1))
        kl_Loss2 = kl_loss(torch.log(F.softmax(nat_logits,dim=1)),F.softmax(teacher_logits_new_nat.detach(),dim=1))
        kl_Loss1 = torch.mean(kl_Loss1)
        kl_Loss2 = torch.mean(kl_Loss2)
        loss = 5.0/6.0*kl_Loss1 + 1.0/6.0*kl_Loss2
        loss.backward()
        optimizer.step()
        if step%100 == 0:
            print('loss',loss.item())
            # break
    if (epoch%10 == 0 and epoch <215) or (epoch%1 == 0 and epoch >= 215):
        test_accs = []
        sum_adv = []
        student.eval()
        for step,(test_batch_data,test_batch_labels) in enumerate(testloader):
            test_batch_data = test_batch_data.float().cuda()
            test_batch_labels = test_batch_labels.cuda()
            test_ifgsm_data = attack_pgd(student,test_batch_data,test_batch_labels,attack_iters=20,step_size=0.003,epsilon=8.0/255.0)
            logits = student(test_ifgsm_data)
            predictions = np.argmax(logits.cpu().detach().numpy(),axis=1)
            predictions = predictions - test_batch_labels.cpu().detach().numpy()
            test_accs = test_accs + predictions.tolist()
            sum_adv.append(np.sum(predictions==0))
        test_accs = np.array(test_accs)
        test_acc = np.sum(test_accs==0)/len(test_accs)
        print('robust acc',np.sum(test_accs==0)/len(test_accs))

        adv = []
        total = []
        for i in range(10):
            adv_i = np.array(sum_adv[i * 10:(i + 1) * 10]).sum() / 10.0
            adv.append(adv_i)
        print("----------class_fairness acc is", adv)

        if test_acc + min(adv)/100.0 > pgd_fair_best_accuracy:
            state = { 'model': student.state_dict(),
                'optimizer': optimizer.state_dict(), 'epoch': epoch}
            torch.save(state,prefix + "_pgd_fair_best"+ '.pth')
            pgd_fair_best_accuracy = test_acc + min(adv)/100.0
            print("best pgd fair accuracy:",str(pgd_fair_best_accuracy))         


        if test_acc> pgd_best_accuracy:
            state = { 'model': student.state_dict(),
                'optimizer': optimizer.state_dict(), 'epoch': epoch}
            torch.save(state, prefix + "_pgd_best"+ '.pth')
            pgd_best_accuracy =test_acc
            print("best pgd accuracy:",str(pgd_best_accuracy))  
        if epoch % 50 == 0:
            state = { 'model': student.state_dict(),
                'optimizer': optimizer.state_dict(), 'epoch': epoch}
            torch.save(state, prefix +"epoch_"+str(epoch)+ '.pth')
    if epoch in [215,260,285]:
        for param_group in optimizer.param_groups:
            param_group['lr'] *= 0.1
            class_labels_temp_rate = class_labels_temp_rate * 0.1
    # update temp
    loss_gap = class_labels_total_loss_adv - torch.mean(class_labels_total_loss_adv)
    class_labels_temp_adv = class_labels_temp_adv - class_labels_temp_rate * loss_gap/ torch.max(torch.abs(loss_gap))
    class_labels_temp_adv = torch.clamp(class_labels_temp_adv, temp_min, temp_max)
    print("class total loss adv")
    print(class_labels_total_loss_adv)

    print("class_labels_temp_adv")
    print(class_labels_temp_adv)

    loss_gap = class_labels_total_loss_nat - torch.mean(class_labels_total_loss_nat)
    class_labels_temp_nat = class_labels_temp_nat - class_labels_temp_rate * loss_gap/ torch.max(torch.abs(loss_gap))
    class_labels_temp_nat = torch.clamp(class_labels_temp_nat, temp_min, temp_max)
    print("class total loss nat")
    print(class_labels_total_loss_nat)    
    print("class_labels_temp_nat")
    print(class_labels_temp_nat)    
