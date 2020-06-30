import glob
import os
import time

import numpy as np
import torch
from torch import nn
from torch.optim import lr_scheduler
from torch.utils import data
from torch.utils.tensorboard import SummaryWriter

from dataset import custom_dataset
from loss import Loss
from model import EAST

try:
    from apex import amp
    GOT_AMP = True
    print("Apex AMP is available!")
except ImportError:
    GOT_AMP = False
    print("Apex AMP is NOT available!")


def train(train_img_path, train_gt_path, pths_path, batch_size, lr, num_workers, epoch_iter, resume):
    os.makedirs(pths_path, exist_ok=True)

    writer = SummaryWriter(os.path.join(pths_path, "logs"))

    file_num = len(os.listdir(train_img_path))

    # Get dataloader
    trainset = custom_dataset(train_img_path, train_gt_path)
    train_loader = data.DataLoader(trainset,
                                   batch_size=batch_size,
                                   shuffle=True,
                                   num_workers=num_workers,
                                   drop_last=True)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    criterion = Loss()

    # Instantiate model
    model = EAST()

    if resume:
        # Resume from latest checkpoint
        checkpoints = glob.glob(pths_path + "/*.pth")
        checkpoints = sorted(checkpoints, key=os.path.getmtime, reverse=True)
        if len(checkpoints) > 0:
            latest_checkpoint = checkpoints[0]
            print(
                f"Restoring model from latest checkpoint: {latest_checkpoint}")
            model.load_state_dict(torch.load(latest_checkpoint,
                                             map_location=torch.device("cpu")))
        else:
            print(
                f"No existing checkpoint found in {pths_path}, training from scratch!")
    else:
        print("Training from scratch!")

    model.to(device)

    # Instantiate optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    # scheduler = lr_scheduler.MultiStepLR(optimizer,
    #                                      milestones=[epoch_iter//2],
    #                                      gamma=0.1)

    # Initialize AMP
    if GOT_AMP:
        model, optimizer = amp.initialize(model, optimizer,
                                          opt_level="O1")

    data_parallel = False
    if torch.cuda.device_count() > 1:
        model = nn.DataParallel(model)
        data_parallel = True
    print(f"Training using {torch.cuda.device_count()} GPUs.")

    # Instantiate scheduler
    scheduler = lr_scheduler.OneCycleLR(optimizer,
                                        max_lr=lr,
                                        total_steps=len(train_loader) * epoch_iter)

    def save_model(step, max_checkpoints=5):
        checkpoints = glob.glob(pths_path + "/*.pth")
        checkpoints = sorted(checkpoints, key=os.path.getmtime, reverse=True)
        if len(checkpoints) >= max_checkpoints:
            for p in checkpoints[max_checkpoints-1:]:
                print(
                    f"Removing checkpoint due to max checkpoint limit ({max_checkpoints}): {p}")
                os.remove(p)

        state_dict = model.module.state_dict() if data_parallel else model.state_dict()
        torch.save(state_dict,
                   os.path.join(
                       pths_path, 'model-{}.pth'.format(step)))

    # Train
    global_step = 0
    for epoch in range(epoch_iter):
        model.train()

        epoch_loss = 0
        epoch_time = time.time()
        for i, (img, gt_score, gt_geo, ignored_map) in enumerate(train_loader):
            start_time = time.time()

            img, gt_score, gt_geo, ignored_map = img.to(device), gt_score.to(
                device), gt_geo.to(device), ignored_map.to(device)
            pred_score, pred_geo = model(img)

            loss = criterion(gt_score, pred_score.to(gt_score.dtype),
                             gt_geo, pred_geo.to(gt_geo.dtype),
                             ignored_map)

            epoch_loss += loss.item()

            optimizer.zero_grad()
            if GOT_AMP:
                with amp.scale_loss(loss, optimizer) as scaled_loss:
                    scaled_loss.backward()
            else:
                loss.backward()
            optimizer.step()
            scheduler.step()

            global_step += 1
            if global_step % 100 == 0:
                print('step : {:7d}, loss: {:.5f}, time: {:.1f} s'.format(
                    global_step,
                    loss.item(),
                    time.time() - start_time,
                ))
                writer.add_scalar("loss/train/step",
                                  loss.item(),
                                  global_step=global_step)
            if global_step % 1000 == 0:
                save_model(global_step)

        # scheduler.step()

        epoch_loss /= int(file_num/batch_size)
        print('epoch: {:4d}, loss: {:.5f}, time {:.1f} s'.format(
            epoch+1,
            epoch_loss,
            time.time()-epoch_time
        ))
        writer.add_scalar("loss/train/epoch",
                          epoch_loss,
                          global_step=epoch)

    # Save final model and exit
    save_model(global_step)
    writer.close()


if __name__ == '__main__':
    train_img_path = os.path.abspath(
        '/mnt/9C5E1A4D5E1A2116/datasets/SynthText/converted/train_img')
    train_gt_path = os.path.abspath(
        '/mnt/9C5E1A4D5E1A2116/datasets/SynthText/converted/train_gt')
    pths_path = './pths4/'
    # train_img_path = os.path.abspath(
    #     '/mnt/9C5E1A4D5E1A2116/datasets/ICDAR_2015/train_img')
    # train_gt_path = os.path.abspath(
    #     '/mnt/9C5E1A4D5E1A2116/datasets/ICDAR_2015/train_gt')
    # pths_path = './ICDAR_2015__cv2minarearect/'
    batch_size = 24
    lr = 3e-4
    num_workers = 16
    epoch_iter = 5

    train(train_img_path, train_gt_path,
          pths_path,
          batch_size,
          lr,
          num_workers,
          epoch_iter,
          resume=True)
