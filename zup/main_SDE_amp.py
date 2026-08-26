import argparse
import os
import time
import torch
import torch.backends.cudnn as cudnn
import torch.optim as optim
from torch.autograd import Variable
from torch.utils.data import DataLoader
import torch.nn.functional as F
from data import Dataset
from SDE import Net_ms2pan_dual,ms2pan_convNet_dual,sobel_filter
from loss import SDE_Losses
import numpy as np
from wald_utilities import wald_protocol_v2
import torch.cuda.amp as amp  # 导入混合精度训练模块
from SDE import sobel_filter

# ================== Pre-Define =================== #
SEED = 10
torch.manual_seed(SEED)
torch.cuda.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)
cudnn.deterministic = True

# ============= HYPER PARAMS(Pre-Defined) ==========#
parser = argparse.ArgumentParser()
parser.add_argument("--lr", type=float, default=0.0005, help="Learning rate")
parser.add_argument("--epochs", type=int, default=100, help="Number of epochs")
parser.add_argument("--batch_size", type=int, default=1, help="Batch size")
parser.add_argument("--device", type=str, default='cuda:0', help="Device to use")
parser.add_argument("--name", type=int,  default=0, help="Data ID (0-19)")
parser.add_argument("--satellite", type=str, default=None, help="Satellite type")
parser.add_argument("--data_path", type=str, default='HardDisk/HeZou/test_wv3_OrigScale_multiExm1.h5', help="Data path")
args = parser.parse_args()


lr = args.lr
epochs = args.epochs
batch_size = args.batch_size
device = torch.device(args.device)
name = args.name
satellite = args.satellite

#model = Net_ms2pan_dual().to(device)
model = ms2pan_convNet_dual().to(device)
optimizer = optim.Adam(model.parameters(), lr=lr, betas=(0.9, 0.999))  # optimizer 1
criterion = SDE_Losses(device)

scaler = amp.GradScaler()  # 创建一个 GradScaler 对象用于自动混合精度


def save_checkpoint(model, name):  # save model_FUG function
    model_out_path = 'model_SDE/' + satellite + '/' + str(name) + '_ms2pan_convNet_dual.pth'
    os.makedirs(os.path.dirname(model_out_path), exist_ok=True)
    torch.save(model.state_dict(), model_out_path)


###################################################################
# ------------------- Main Train (Run second)----------------------------------
###################################################################


def train(training_data_loader, name):
    t1 = time.time() # training time
    print('Run SDE...')
    min_loss = 1
    for epoch in range(epochs):
        epoch += 1
        epoch_train_loss, epoch_val_loss = [], []

        # ============Epoch Train=============== #
        model.train()

        for iteration, batch in enumerate(training_data_loader, 1):
            ms, lms, pan = Variable(batch[0]).to(device), \
                      Variable(batch[1]).to(device), \
                      Variable(batch[2], requires_grad=False).to(device)

            pan = wald_protocol_v2(ms, pan, 4, sensor=satellite).to(device)
            #pan = F.max_pool2d(pan, kernel_size=4, stride=4)  # 使用最大池化代替Wald协议,效果略差一点点
 
            optimizer.zero_grad()  # fixed
            pan_gra_x,pan_gra_y = sobel_filter(pan)  # compute gradient of pan
            ms_gra_x,ms_gra_y = sobel_filter(ms)  # compute gradient of ms

            with amp.autocast():  # 使用自动混合精度
                out1,out2 = model(ms_gra_x,ms_gra_y)

                loss_x = criterion(out1, pan_gra_x)  # compute loss
                loss_y = criterion(out2, pan_gra_y)  # compute loss
                loss = (loss_x + loss_y)/100  # compute total loss

            scaler.scale(loss).backward()  # 混合精度梯度计算
            scaler.step(optimizer)  # 混合精度优化
            scaler.update()  # 更新 scaler

            epoch_train_loss.append(loss.item())  # save all losses into a vector for one epoch

        t_loss = np.nanmean(np.array(epoch_train_loss))
        if t_loss < min_loss:
            save_checkpoint(model, name)
            min_loss = t_loss
        if epoch % 20 == 0:
            print('SDE stage: Epoch: {} training loss: {:.7f}'.format(epoch, t_loss))
    t2 = time.time()  # training time
    print(f'SDE time: {t2-t1}s')  # training time
###################################################################
# ------------------- Main Function (Run first) -------------------
###################################################################


if __name__ == "__main__":
    train_set = Dataset(args.data_path, name)  # put training data to Dataset for batches
    training_data_loader = DataLoader(dataset=train_set, num_workers=0, batch_size=batch_size, shuffle=True,
                                      pin_memory=True,
                                      drop_last=True)  # put training data to DataLoader for batches
    train(training_data_loader, name)
