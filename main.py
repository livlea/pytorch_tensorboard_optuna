import os
from datetime import datetime
import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import optuna

from model import network
from utils.utils import str2bool, AverageMeter


parser = argparse.ArgumentParser(description='Classification test.')
parser.add_argument('-b', '--batch_size', default=32, type=int, help='Number of batch size (default: 32)')
parser.add_argument('-j', '--workers', default=4, type=int, help='Number of data loading workers (default: 4)')
parser.add_argument('-e', '--epochs', default=10, type=int, help='Number of epochs (default: 10)')
parser.add_argument('-s', '--show_image', default=False, type=str2bool, help='Show images (default: False)')
parser.add_argument('-p', '--path_of_results', default='./results', type=str, help='Path of results, net will be saved (default: ./results)')
parser.add_argument('-t', '--tensorboard', default=True, type=str2bool, help='Dump logs to tensorboard (default: True)')
parser.add_argument('-o', '--optuna', default=True, type=str2bool, help='Run hyperparameter tuning by optuna (default: True)')
parser.add_argument('--optuna_trialnum', default=20, type=int, help='Number of trials of optuna (default: 20)')
args = parser.parse_args()

classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

trial_num = -1 # trial number of optuna
best_accuracy = 0.0 # best valid accuracy of optuna trials
best_num = -1 # best trial number of optuna

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def main():
    if not os.path.isdir(args.path_of_results):
        os.makedirs(args.path_of_results)

    if args.tensorboard:
        current_time = datetime.now().strftime('%b%d_%Y_%Hh%Mm%Ss')
        log_dir = os.path.join('runs', current_time)

        # Init SummaryWriter of tensorboard. Writer will ouput to ./runs/ directory by default.
        writer = SummaryWriter(log_dir=log_dir)

        # Add args states to writer.
        writer.add_text('args', str(args), 0)

    writer = writer if args.tensorboard else None

    transform = transforms.Compose(
        [transforms.ToTensor(),
         transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

    trainvalidset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
    n_samples = len(trainvalidset) # n_samples is 50000 (CIFAR10)
    train_size = int(len(trainvalidset) * 0.8) # train_size is 40000 (CIFAR10)
    valid_size = n_samples - train_size # valid_size is 10000 (CIFAR10)

    trainset, validset = torch.utils.data.random_split(trainvalidset, [train_size, valid_size])
    testset  = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=args.batch_size, shuffle=True, num_workers=args.workers)
    validloader = torch.utils.data.DataLoader(validset, batch_size=args.batch_size, shuffle=False, num_workers=args.workers)
    testloader  = torch.utils.data.DataLoader(testset, batch_size=args.batch_size, shuffle=False, num_workers=args.workers)

    # check train and test images
    checkimages(trainloader, writer, mode='train')
    checkimages(validloader, writer, mode='valid')
    checkimages(testloader, writer, mode='test')

    if args.optuna:
        # Hyperparameter tuning by using optuna
        study = optuna.create_study(direction='maximize')
        study.optimize(objective_variable(trainloader, validloader, writer), n_trials=args.optuna_trialnum)
        print('Best params : {}'.format(study.best_params))
        print('Best value  : {}'.format(study.best_value))
        print('Best trial  : {}'.format(study.best_trial))

        df = study.trials_dataframe()
        print(df)

        if args.tensorboard:
            df_records = df.to_dict(orient='records')

            for i in range(len(df_records)):
                df_records[i]['datetime_start'] = str(df_records[i]['datetime_start'])
                df_records[i]['datetime_complete'] = str(df_records[i]['datetime_complete'])
                value = df_records[i].pop('value')
                value_dict = {'value': value}
                writer.add_hparams(df_records[i], value_dict)

    else:
        # Train
        net = network.Net().to(device)
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)
        train(net, trainloader, validloader, optimizer, criterion, writer, trial_num)

    # Evaluate best net using test data
    best_net = network.Net().to(device)
    best_net.load_state_dict(torch.load(args.path_of_results + '/best_net_{:03d}'.format(best_num) + '.pth.tar'))

    # Read test data
    dataiter = iter(testloader)
    images, labels = dataiter.next()
    del dataiter

    # Compare groundtruth to predicted
    outputs = best_net(images.to(device))
    _, predicted = torch.max(outputs, 1)
    print('GroundTruth: ', ' '.join('%5s' % classes[labels[j]] for j in range(args.batch_size)))
    print('Predicted: ', ' '.join('%5s' % classes[predicted[j]] for j in range(args.batch_size)))

    # Evaluate
    test_accuracy, test_accuracy_of_classes = evaluate(best_net, testloader, 'test', best_num)
    print('Test accuracy of best net : {:.3f}'.format(test_accuracy))

    if args.tensorboard:
        writer.add_scalars('Accuracy/test/all', {'trial_{:03d}'.format(best_num): test_accuracy}, best_num)

        for i in range(10):
            writer.add_scalars('Accuracy/test/classes', {'trial_{:03d}'.format(trial_num): test_accuracy_of_classes[i]}, i)

        writer.flush()
        writer.close()


def checkimages(loader, writer, mode='train'):
    dataiter = iter(loader)
    images, labels = dataiter.next()

    if args.show_image:
        imshow(torchvision.utils.make_grid(images))

    print(' '.join('%5s' % classes[labels[j]] for j in range(images.size(0))))

    if args.tensorboard:
        grid = torchvision.utils.make_grid(images)
        writer.add_image('Images/{}'.format(mode), grid/2 + 0.5, 0)

        if mode == 'train':
            net = network.Net().to(device)
            writer.add_graph(net, images.to(device))

    del dataiter


def train(net, trainloader, validloader, optimizer, criterion, writer, trial_num):
    best_valid_accuracy = 0.0
    print('trial id : {}'.format(trial_num))

    for epoch in range(args.epochs):  # loop over the dataset multiple times
        train_loss = AverageMeter()
        valid_loss = AverageMeter()

        for i, data in enumerate(trainloader, 0):
            # get the inputs; data is a list of [inputs, labels]
            inputs, labels = data

            # zero the parameter gradients
            optimizer.zero_grad()

            # forward + backward + optimize
            outputs = net(inputs.to(device))
            loss = criterion(outputs, labels.to(device))
            loss.backward()
            optimizer.step()

            train_loss.update(loss.item(), args.batch_size)

        torch.save(net.state_dict(), args.path_of_results + '/checkpoint.pth.tar')

        for i, data in enumerate(validloader, 0):
            inputs, labels = data
            outputs = net(inputs.to(device))
            loss = criterion(outputs, labels.to(device))
            valid_loss.update(loss.item(), args.batch_size)

        valid_accuracy, valid_accuracy_of_classes = evaluate(net, validloader, 'valid', trial_num)

        if valid_accuracy >= best_valid_accuracy:
            best_valid_accuracy = valid_accuracy

            # save best trained params
            print('save best net at trial_num {}'.format(trial_num))
            torch.save(net.state_dict(), args.path_of_results + '/best_net_{:03}'.format(trial_num) + '.pth.tar')

        print('[{:03d} / {:03d}] train_loss, valid_loss, valid_accuracy : {:.3f}, {:.3f}, {:.3f}'.format(epoch + 1, args.epochs, train_loss.avg, valid_loss.avg, valid_accuracy))

        if args.tensorboard:
            # writer.add_scalar('Loss/train', train_loss.avg, epoch)
            writer.add_scalars('Loss/train', {'trial_{:03d}'.format(trial_num): train_loss.avg}, epoch)
            writer.add_scalars('Loss/valid', {'trial_{:03d}'.format(trial_num): valid_loss.avg}, epoch)
            writer.add_scalars('Accuracy/valid', {'trial_{:03d}'.format(trial_num): valid_accuracy}, epoch)

            writer.flush()

    if args.tensorboard:
        writer.add_scalars('Accuracy/valid/all', {'trial_{:03d}'.format(trial_num): valid_accuracy}, trial_num)

        for i in range(10):
            writer.add_scalars('Accuracy/valid/classes', {'trial_{:03d}'.format(trial_num): valid_accuracy_of_classes[i]}, i)

        writer.flush()

    global best_accuracy
    if best_valid_accuracy >= best_accuracy:
        best_accuracy = best_valid_accuracy
        global best_num
        best_num = trial_num

    print('best_valid_accuracy of this trial: {:.3f}'.format(best_valid_accuracy))
    print('best_accuracy of trials : {:.3f}'.format(best_accuracy))
    print('best_num of trials: {:.3f}'.format(best_num))
    print('Finished Training')

    return best_valid_accuracy


def evaluate(net, dataloader, mode, trial_num):
    correct = 0
    total = 0
    class_correct = list(0. for i in range(10))
    class_total = list(0. for i in range(10))

    with torch.no_grad():
        for data in dataloader:
            images, labels = data
            outputs = net(images.to(device))
            _, predicted = torch.max(outputs.to('cpu').data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            c = (predicted == labels).squeeze()
            for i in range(4):
                label = labels[i]
                class_correct[label] += c[i].item()
                class_total[label] += 1

    accuracy = 100 * correct / total
    print('Accuracy of the network : {:.3f}'.format(accuracy))

    accuracy_of_classes = []
    for i in range(10):
        accuracy_of_class = 100 * class_correct[i] / class_total[i]
        accuracy_of_classes.append(accuracy_of_class)
        print('Accuracy of {} : {:.3f}'.format(classes[i], accuracy_of_class))

    return accuracy, accuracy_of_classes


def imshow(img):
    img = img / 2 + 0.5     # unnormalize
    npimg = img.numpy()
    plt.imshow(np.transpose(npimg, (1, 2, 0)))
    plt.show()


def get_optimizer(trial, model):
    # Search Adam and SGD
    optimizer_names = ['Adam', 'SGD']
    optimizer_name = trial.suggest_categorical('optimizer', optimizer_names)

    weight_decay = trial.suggest_loguniform('weight_decay', 1e-10, 1e-3)
    lr = trial.suggest_loguniform('lr', 1e-5, 1e-1)

    if optimizer_name == optimizer_names[0]:
        optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    else:
        optimizer = optim.SGD(model.parameters(), lr=lr, weight_decay=weight_decay, momentum=0.9)
    return optimizer


def objective_variable(trainloader, validloader, writer):
    def objective(trial):
        global trial_num
        trial_num += 1

        model = network.Net().to(device)
        optimizer = get_optimizer(trial, model)
        criterion = nn.CrossEntropyLoss()

        # Training
        valid_accuracy = train(model, trainloader, validloader, optimizer, criterion, writer, trial_num)

        # Hyperparameter tuning will be done as return become max, since this code use direction='maximize'.
        return valid_accuracy

    return objective


if __name__  == '__main__':
    main()
