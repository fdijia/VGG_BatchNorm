import json
from os.path import exists

import matplotlib as mpl
from matplotlib.pyplot import tight_layout

mpl.use('Agg')
import matplotlib.pyplot as plt
from torch import nn
import numpy as np
import torch
import os
import random
from tqdm import tqdm as tqdm
from IPython import display

from models.vgg import VGG_A
from models.vgg import VGG_A_BatchNorm
from data.loaders import get_cifar_loader


num_workers = 4
batch_size = 100
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

"""获取数据集"""
def get_dataloader():
    train = get_cifar_loader(batch_size=batch_size, train=True, num_workers=num_workers)
    test = get_cifar_loader(batch_size=batch_size, train=False, num_workers=num_workers)
    return train, test

"""训练模型"""
def train(model, optimizer, criterion, train_loader, val_loader, scheduler=None, epochs_n=80, save_dir=None):
    model.to(device)
    learning_curve = [np.nan] * epochs_n
    train_accuracy_curve = [np.nan] * epochs_n
    val_accuracy_curve = [np.nan] * epochs_n
    max_val_accuracy = 0
    max_val_accuracy_epoch = 0

    batches_n = len(train_loader)
    losses_list = []
    grads = []
    deta_grads = []
    g_old = None
    for epoch in tqdm(range(epochs_n), unit='epoch'):
        # Training phase
        model.train()
        loss_list = []  # use this to record the loss value of each step
        grad = []  # use this to record the loss gradient of each step
        deta_grad = []
        learning_curve[epoch] = 0  # maintain this to plot the training curve

        train_correct = 0
        train_total = 0

        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            prediction = model(x)
            loss = criterion(prediction, y)

            loss.backward()
            loss_list.append(loss.item())
            g = model.classifier[4].weight.grad.clone()
            if g_old:
                deta_g = g - g_old
            else:
                deta_g = g
            g_old = g
            grad.append(g.norm().item())
            deta_grad.append(deta_g.norm().item())

            optimizer.step()
            learning_curve[epoch] += loss.item()

            # Calculate training accuracy
            _, predicted = torch.max(prediction.data, 1)
            train_total += y.size(0)
            train_correct += (predicted == y).sum().item()

        if scheduler is not None:
            scheduler.step()

        losses_list.append(loss_list)
        grads.append(grad)
        deta_grads.append(deta_grad)
        learning_curve[epoch] /= batches_n
        train_accuracy_curve[epoch] = 100 * train_correct / train_total

        # Validation phase
        val_accuracy_curve[epoch] = get_accuracy(model, val_loader)

        # Save best model if specified
        if val_accuracy_curve[epoch] > max_val_accuracy:
            max_val_accuracy = val_accuracy_curve[epoch]
            max_val_accuracy_epoch = epoch
            if save_dir is not None:
                os.makedirs(save_dir, exist_ok=True)
                torch.save(model.state_dict(), save_dir+'/best.pth')

        # Print epoch summary
        print(f'Epoch [{epoch + 1}/{epochs_n}] - '
              f'Train Loss: {learning_curve[epoch]:.4f}, '
              f'Train Acc: {train_accuracy_curve[epoch]:.2f}%, '
              f'Val Acc: {val_accuracy_curve[epoch]:.2f}%')

    print(f'Best validation accuracy: {max_val_accuracy:.2f}% at epoch {max_val_accuracy_epoch + 1}')

    # Return comprehensive training results
    training_results = {
        'learning_curve': learning_curve,   # average loss per epoch
        'train_accuracy_curve': train_accuracy_curve,   # train accuracy per epoch
        'val_accuracy_curve': val_accuracy_curve,   # val accuracy per epoch
        'losses_list': losses_list, # list of l list loss per step
        'grads': grads, # list of a list grad per step
        'deta_grads': deta_grads, # list of a list deta grad per step
        'max_val_accuracy': max_val_accuracy,
        'max_val_accuracy_epoch': max_val_accuracy_epoch + 1
    }

    if save_dir:
        with open(save_dir + '/results.json', 'w') as f:
            json.dump(training_results, f)

    return training_results

"""测试模型"""
def get_accuracy(model, data_loader):
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for x, y in data_loader:
            x, y = x.to(device), y.to(device)
            outputs = model(x)
            _, predicted = torch.max(outputs.data, 1)
            total += y.size(0)
            correct += (predicted == y).sum().item()
    return 100 * correct / total

"""设置随机数种子，使实验能够复现"""
def set_random_seeds(seed_value=0, device='cpu'):
    np.random.seed(seed_value)
    torch.manual_seed(seed_value)
    random.seed(seed_value)
    if device != 'cpu':
        torch.cuda.manual_seed(seed_value)
        torch.cuda.manual_seed_all(seed_value)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

"""对比两个模型训练过程中的训练损失、训练准确率、测试准确率、loss梯度"""
def compare_training_results(results_without_bn, results_with_bn, save_path=None):
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))

    epochs = range(1, len(results_without_bn['learning_curve']) + 1)

    # Plot training loss comparison
    axes[0, 0].plot(epochs, results_without_bn['learning_curve'],
                    label='VGG-A without BN', color='blue', linewidth=2)
    axes[0, 0].plot(epochs, results_with_bn['learning_curve'],
                    label='VGG-A with BN', color='red', linewidth=2)
    axes[0, 0].set_title('Training Loss Comparison')
    axes[0, 0].set_xlabel('Epoch')
    axes[0, 0].set_ylabel('Loss')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)

    # Plot training accuracy comparison
    axes[0, 1].plot(epochs, results_without_bn['train_accuracy_curve'],
                    label='VGG-A without BN', color='blue', linewidth=2)
    axes[0, 1].plot(epochs, results_with_bn['train_accuracy_curve'],
                    label='VGG-A with BN', color='red', linewidth=2)
    axes[0, 1].set_title('Training Accuracy Comparison')
    axes[0, 1].set_xlabel('Epoch')
    axes[0, 1].set_ylabel('Accuracy (%)')
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)

    # Plot validation accuracy comparison
    axes[0, 2].plot(epochs, results_without_bn['val_accuracy_curve'],
                    label='VGG-A without BN', color='blue', linewidth=2)
    axes[0, 2].plot(epochs, results_with_bn['val_accuracy_curve'],
                    label='VGG-A with BN', color='red', linewidth=2)
    axes[0, 2].set_title('Validation Accuracy Comparison')
    axes[0, 2].set_xlabel('Epoch')
    axes[0, 2].set_ylabel('Accuracy (%)')
    axes[0, 2].legend()
    axes[0, 2].grid(True, alpha=0.3)

    # Plot gradient norm comparison
    grads_with_bn, grad_without_bn = results_with_bn['grads'], results_without_bn['grads']
    changes_with_bn, changes_without_bn = results_with_bn['deta_grads'], results_without_bn['deta_grads']

    avg_grads_without_bn = [np.mean(g) for g in grad_without_bn]
    avg_grads_with_bn = [np.mean(g) for g in grads_with_bn]
    avg_change_with_bn = [np.mean(g) for g in changes_with_bn]
    avg_change_without_bn = [np.mean(g) for g in changes_without_bn]

    axes[1, 0].plot(epochs, avg_grads_without_bn,
                    label='VGG-A without BN Grad', color='blue', linewidth=2)
    axes[1, 0].plot(epochs, avg_grads_with_bn,
                    label='VGG-A with BN Grad', color='red', linewidth=2)
    axes[1, 0].plot(epochs, avg_change_without_bn,
                    label='VGG-A without BN ΔGrad', color='green', linewidth=2)
    axes[1, 0].plot(epochs, avg_change_with_bn,
                    label='VGG-A with BN ΔGrad', color='purple', linewidth=2)

    axes[1, 0].set_title('Average Gradient Norm Comparison')
    axes[1, 0].set_xlabel('Epoch')
    axes[1, 0].set_ylabel('Gradient Norm and Gradient Change Norm')
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)

    # Plot Max difference in gradient over a distance of 50
    deta_grads_bn, deta_grads_notbn = [], []
    for i in range(len(changes_without_bn)):
        deta_grads_bn.extend(changes_with_bn[i])
        deta_grads_notbn.extend(changes_without_bn[i])
    deta_grads_bn = [max(deta_grads_bn[i*50:(i+1)*50]) for i in range(len(deta_grads_bn)//50)]
    deta_grads_notbn = [max(deta_grads_notbn[i*50:(i+1)*50]) for i in range(len(deta_grads_notbn)//50)]

    axes[1, 1].plot(range(1, len(deta_grads_notbn)+1), deta_grads_notbn,
                    label='VGG-A without BN', color='blue', linewidth=2)
    axes[1, 1].plot(range(1, len(deta_grads_bn)+1), deta_grads_bn,
                    label='VGG-A with BN', color='red', linewidth=2)
    axes[1, 1].set_title('Max difference in gradient over a distance of 50')
    axes[1, 1].set_xlabel('Steps')
    axes[1, 1].set_ylabel('Gradient Change')
    axes[1, 1].legend()
    axes[1, 1].grid(True, alpha=0.3)

    # 结果
    axes[1, 2].text(0.5, 0.3, f'With BN Val Accuracy: {results_with_bn['max_val_accuracy']}%',
                    ha='center', va='center', fontsize=12)

    axes[1, 2].text(0.5, 0.7, f'Without BN Val Accuracy: {results_without_bn['max_val_accuracy']}%',
                    ha='center', va='center', fontsize=12)
    axes[1, 2].axis('off')

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    else:
        plt.show()
    plt.close()

"""根据特定的学习率对两个模型进行训练，如果以及训练过（存在训练历史结果）则直接读取结果"""
def get_results(lr, criterion, train_loader, val_loader, epo=80, milestones=None):
    if milestones is None:
        milestones = [40, 65, 78]
    base_dir = 'save_models'
    vgga_path = base_dir + '/vgga' + str(lr)
    print('='*50, '\nstart training vgg, lr = ', lr, '\n', '='*50)
    if not os.path.exists(vgga_path + '/results.json'):
        vgga = VGG_A()
        optimizer = torch.optim.Adam(vgga.parameters(), lr=lr)
        scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=milestones, gamma=0.2)
        vgga_results = train(vgga, optimizer, criterion, train_loader, val_loader, scheduler, epo, vgga_path)
    else:
        with open(vgga_path + '/results.json', 'r') as f:
            vgga_results = json.load(f)

    vggabn_path = base_dir + '/vggabn' + str(lr)
    print('='*50, '\nstart training vgg with bn, lr = ', lr, '\n', '='*50)
    if not os.path.exists(vggabn_path + '/results.json'):
        vggabn = VGG_A_BatchNorm()
        optimizer = torch.optim.Adam(vggabn.parameters(), lr=lr)
        scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[epo//2, epo//4*3, epo//10*9], gamma=0.2)
        vggabn_results = train(vggabn, optimizer, criterion, train_loader, val_loader, scheduler, epo, vggabn_path)
    else:
        with open(vggabn_path + '/results.json', 'r') as f:
            vggabn_results = json.load(f)

    return vgga_results, vggabn_results

"""给定一列学习率，对每个学习率都训练两个模型，并返回包含两个模型的训练历史的列表（每个模型一个列表）
调用过程中会自动对每个学习率画两个模型对比图"""
def get_history(lrs, criterion, train_loader, val_loader, epo=80, milestones=None):
    bn_curves, notbn_curves = [], []

    for lr in lrs:
        vgga, vggabn = get_results(
            lr, criterion, train_loader, val_loader, epo=epo, milestones=milestones
        )
        compare_training_results(vgga, vggabn, save_path='comparison/lr'+str(lr)+'.png')
        notbn_curves.append(vgga)
        bn_curves.append(vggabn) # epoch loss -> epochs -> models
    return bn_curves, notbn_curves

"""从get_history中获取的历史记录中获取对应名字的列表
如获取losses list时，分别获取两个模型每个学习率的losses list并返回两个列表
同时返回的列表进行切片操作，防止数据太密集"""
def get_landscape(bn, notbn, steps, value=None):
    if value:
        bn = [d[value] for d in bn]
        notbn = [d[value] for d in notbn]

    min_curve, max_curve = [[], []], [[], []]   # 第一个不带bn，第二个带bn
    epochs, lengths = len(bn[0]), len(bn[0][0])
    for e in range(epochs):
        for i in range(lengths):
            if e * lengths + i >= steps[1]:
                break
            if e * lengths + i < steps[0]:
                continue
            bn_curve = [l[e][i] for l in bn]
            notbn_curve = [l[e][i] for l in notbn]
            min_curve[0].append(min(notbn_curve))
            max_curve[0].append(max(notbn_curve))
            min_curve[1].append(min(bn_curve))
            max_curve[1].append(max(bn_curve))

    return min_curve, max_curve

"""从get landscape中获取的单个属性的记录绘制landscape"""
def plot_loss_landscape(min_curves, max_curves, title, interval=15, figures_path='comparison'):
    epochs = range(len(min_curves[0]))
    plt.figure(figsize=(15, 5))

    # 绘制不带BN的曲线范围
    plt.fill_between(
        epochs[::interval],
        min_curves[0][::interval],
        max_curves[0][::interval],
        color='lightblue',
        edgecolor='darkblue',
        alpha=0.6,
        linewidth=1.5,
        label='Without BN Range'
    )

    # 绘制带BN的曲线范围
    plt.fill_between(
        epochs[::interval],
        min_curves[1][::interval],
        max_curves[1][::interval],
        color='moccasin',
        edgecolor='darkorange',
        alpha=0.6,
        linewidth=1.5,
        label='With BN Range'
    )

    plt.xlabel("Epoch")
    plt.ylabel(title)
    plt.title(title + " Landscape Comparison: With vs Without BatchNorm")
    plt.legend()
    plt.grid(True)

    # 保存图像（需提前定义 figures_path）
    plt.savefig(figures_path + '/' + title + ' landscape.png')
    plt.close()

"""此处函数为绘制loss landscape的接口，bn、notbn为训练结果字典的列表或者某种记录的列表"""
def compare_loss_landscape(bn, notbn, steps=(0, 10000), interval=15, value=None, fig_title='Loss'):
    assert steps[0] < steps[1]

    min_curve, max_curve = get_landscape(bn, notbn, steps, value)
    plot_loss_landscape(min_curve, max_curve, fig_title, interval=interval)


"""主函数"""
def main():
    os.makedirs('save_models', exist_ok=True)
    os.makedirs('comparison', exist_ok=True)
    train, val = get_dataloader()
    lrs = [1e-4, 2.5e-4, 5e-4, 1e-3]
    set_random_seeds(seed_value=2020, device=device)
    criterion = nn.CrossEntropyLoss()
    bn, notbn = get_history(lrs, criterion, train, val, epo=27, milestones=[12, 20, 25]) # 训练模型并得到训练记录
    compare_loss_landscape(bn, notbn, (0, 13500), interval=30, value='losses_list', fig_title='Loss') # 通过训练记录画出loss landscape
    compare_loss_landscape(bn, notbn, (0, 13500), interval=30, value='deta_grads', fig_title='Grad Change')


if __name__ == "__main__":
    main()
