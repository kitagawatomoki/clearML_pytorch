import torch
import torchmetrics
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter
import argparse
import os
import time
import gc
import numpy as np
import yaml
from tqdm import tqdm
import sys

from model import get_model
from dataset import Dataset

from clearml import Task

def main():
    Task.set_random_seed(7)

    args = create_argparser().parse_args()
    device = torch.device('cuda:%d'%(args.gpu) if torch.cuda.is_available() else 'cpu')

    task = Task.init(project_name='clearLM_sweep'
                    ,task_name='Pytorch cifar base'
                    ,auto_connect_arg_parser=False
                    ,auto_connect_frameworks={'pytorch':False})
    task.set_script(working_dir="/Work30/kitagawatomoki/pytorch_cifar10"
                    ,entry_point="base_template_cifar10.py")

    config = {"device":device
            ,"model":"resnet18"
            ,"optimizer":"Adam"
            ,"aug":False
            ,"lr":0.0001
            ,"epochs":1}

    model,batch_size = get_model(config['model'], 10)
    model.to(device)
    config['mean'] = model.default_cfg["mean"]
    config['std'] = model.default_cfg["std"]
    config['size'] = model.default_cfg["input_size"][1]
    config['batch'] = batch_size

    run_name = "model-{}_size-{}_batch-{}_aug-{}".format(config['model']
                                                        ,config['size']
                                                        ,config['batch']
                                                        ,config['aug'])
    save_log = os.path.join("experiment", run_name, 'log')
    save_model = os.path.join("experiment", run_name, 'model')
    os.makedirs(save_log, exist_ok=True)
    os.makedirs(save_model, exist_ok=True)
    summary_writer = SummaryWriter(log_dir=save_log)


    config = task.connect(config)

    train_dataset = Dataset(config,"train")
    test_dataset = Dataset(config ,"test")

    train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=2)
    test_dataloader = torch.utils.data.DataLoader(test_dataset, batch_size=int(batch_size/4), shuffle=False, num_workers=2)

    if config['optimizer']=="SGD":
        optimizer = torch.optim.SGD(model.parameters(), lr=config['lr'])
    elif config['optimizer']=="Adam":
        optimizer = torch.optim.Adam(model.parameters(), lr=config['lr'])
    elif config['optimizer']=="AdamW":
        optimizer = torch.optim.AdamW(model.parameters(), lr=config['lr'])

    optimizer = torch.optim.Adam(model.parameters(), lr=config['lr'])
    metric = torchmetrics.Accuracy(task="multiclass", num_classes=10)
    criterion = nn.CrossEntropyLoss()

    train_step = 0
    test_step = 0
    best_acc = 0

    for epoch in range(config['epochs']):
        model.train()#学習モードに移行
        s_time = time.time()

        for batch in tqdm(train_dataloader):
            images, labels = batch
            images, labels = images.to(device), labels.to(device)

            optimizer.zero_grad()
            preds = model(images)

            loss = criterion(preds, labels)

            summary_writer.add_scalar("train_loss", float(loss.to('cpu').detach().numpy().copy()), train_step)

            preds = preds.softmax(dim=-1)
            preds = preds.to('cpu').detach()
            labels = labels.to('cpu').detach()
            acc = metric(preds, labels)
            summary_writer.add_scalar("train_acc", acc, train_step)

            loss.backward()
            optimizer.step()
            train_step+=1

        model.eval()#学習モードに移行
        test_acc = 0
        test_loss = 0
        for i, batch in enumerate(test_dataloader):
            images, labels = batch
            images, labels = images.to(device), labels.to(device)

            preds = model(images)

            loss = criterion(preds, labels)

            summary_writer.add_scalar("test_loss", float(loss.to('cpu').detach().numpy().copy()), test_step)

            preds = preds.softmax(dim=-1)
            preds = preds.to('cpu').detach()
            labels = labels.to('cpu').detach()
            acc = metric(preds, labels)
            summary_writer.add_scalar("test_acc", acc, test_step)
            test_step+=1

            test_acc+=acc
            test_loss+=float(loss.to('cpu').detach().numpy().copy())

        test_acc/=len(test_dataloader)
        test_loss/=len(test_dataloader)
        summary_writer.add_scalar("acc", test_acc, epoch)
        summary_writer.add_scalar("loss", test_loss, epoch)

        print('Time for epoch {} is {} | Val Acc {}'.format(epoch+1, time.time() - s_time, test_acc))

        if test_acc >=best_acc:
            best_acc = test_acc
            torch.save(model.state_dict(), os.path.join(save_model, "best_classifier_model.pt"))

        if epoch % 5 == 0:
            torch.save(model.state_dict(), os.path.join(save_model, "classifier_model_ep{}.pt".format(epoch)))

    torch.save(model.state_dict(), os.path.join(save_model, "classifier_model.pt"))

def create_argparser():
    parser = argparse.ArgumentParser()
    parser.add_argument("-g", "--gpu", type=int, default=0)

    return parser

if __name__ == "__main__":
    main()