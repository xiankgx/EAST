import argparse
import glob
import math
import os
import time
from shutil import copyfile, copytree, rmtree

import numpy as np
import torch
import yaml
from torch import nn
from torch.optim import lr_scheduler
from torch.utils import data
from torch.utils.tensorboard import SummaryWriter

from dataset import custom_dataset
from loss import Loss
from models import EAST


class Trainer(object):
    """An object that can be used for training a EAST model."""

    def __init__(self, config_path: str):
        """Create a trainer object.

        Args:
            config_path (str): configuration file path
        """

        self.config_path = config_path
        self._parse_config()

        os.makedirs(self.config["training"]["prefix"], exist_ok=True)
        self._save_code()

    def _parse_config(self):
        """Parse configurations from config file."""
        with open(self.config_path, "r") as f:
            self.config = yaml.load(f, Loader=yaml.Loader)

    def _save_code(self):
        """Copy vital source code files and folders to a directory."""

        outdir = os.path.join(self.config["training"]["prefix"], "code")
        os.makedirs(outdir, exist_ok=True)

        backup_list = [
            "dataset.py",
            "detect.py",
            "eval.py",
            "loss.py",
            "models",
            "train.py",
            "configs",
        ]
        for p in backup_list:
            dst_path = os.path.join(outdir, p)
            # Remove existing file or folders
            if os.path.exists(dst_path):
                if os.path.isfile(dst_path):
                    os.remove(dst_path)
                elif os.path.isdir(dst_path):
                    rmtree(dst_path)

            if os.path.isfile(p):
                copyfile(p, dst_path)
            elif os.path.isdir(p):
                copytree(p, dst_path)

    def train(self):
        """Train a model using configurations from the configuration file."""

        # Writer for TensorBoard logging
        writer = SummaryWriter(os.path.join(self.config["training"]["prefix"],
                                            "logs"))

        file_num = len(os.listdir(
            self.config["training"]["img_root_dir"]["train"]))

        device = "cuda" if torch.cuda.is_available() else "cpu"

        # The loss function for training the model
        criterion = Loss()

        # Instantiate model
        model = EAST(**self.config["model"])

        # Restore model weights from checkpoint
        checkpoints_dir = os.path.join(self.config["training"]["prefix"],
                                       "checkpoints")
        if self.config["training"]["resume"]:
            # Resume from latest checkpoint
            checkpoints = glob.glob(checkpoints_dir + "/*.pth")
            checkpoints = sorted(checkpoints,
                                 key=os.path.getmtime,
                                 reverse=True)
            if len(checkpoints) > 0:
                latest_checkpoint = checkpoints[0]
                print(
                    f"Restoring model from latest checkpoint: {latest_checkpoint}")

                try:
                    model.load_state_dict(torch.load(latest_checkpoint,
                                                     map_location=torch.device("cpu")))
                except:
                    print(
                        f"Failed to restore from checkpoint: {latest_checkpoint}")
            else:
                print(
                    f"No existing checkpoint found in {checkpoints_dir}, training from scratch!")
        else:
            print("Training from scratch!")

        # Move model to the right device
        model.to(device)

        # Instantiate optimizer
        optimizer = torch.optim.Adam(model.parameters(),
                                     lr=self.config["training"]["learning_rate"])
        # scheduler = lr_scheduler.MultiStepLR(optimizer,
        #                                      milestones=[epoch_iter//2],
        #                                      gamma=0.1)

        # Initialize AMP
        if GOT_AMP and self.config["training"]["mixed_precision"]:
            model, optimizer = amp.initialize(model, optimizer,
                                              opt_level="O1")

        data_parallel = False
        if self.config["training"]["multi_gpu"] and torch.cuda.device_count() > 1:
            model = nn.DataParallel(model)
            data_parallel = True
            print(f"Training using {torch.cuda.device_count()} GPUs.")
        else:
            print(f"Training using 1 {'GPU' if 'cuda' in  device else 'CPU'}.")

        dummy_out, _ = model(torch.rand(4, 3, self.config["model"]["scope"], self.config["model"]["scope"],
                                        device=device))
        scale = dummy_out.size(2)/self.config["model"]["scope"]
        print(f"scale: {scale}")

        # Get dataloader
        normalization_params = model.module.get_preprocessing_params() if hasattr(
            model, "module") else model.get_preprocessing_params()
        trainset = custom_dataset(img_path=self.config["training"]["img_root_dir"]["train"],
                                  gt_path=self.config["training"]["annotations_root_dir"]["train"],
                                  scale=scale,
                                  normalization_params=normalization_params,
                                  length=self.config["model"]["scope"]
                                  )
        train_loader = data.DataLoader(trainset,
                                       batch_size=self.config["training"]["batch_size"],
                                       shuffle=True,
                                       num_workers=self.config["training"]["num_workers"],
                                       drop_last=True)

        test_loader = None
        if "test" in self.config["training"]["img_root_dir"]:
            testset = custom_dataset(img_path=self.config["training"]["img_root_dir"]["test"],
                                     gt_path=self.config["training"]["annotations_root_dir"]["test"],
                                     scale=scale,
                                     normalization_params=normalization_params,
                                     length=self.config["model"]["scope"]
                                     )
            test_loader = data.DataLoader(testset,
                                          batch_size=self.config["training"]["batch_size"],
                                          shuffle=True,
                                          num_workers=self.config["training"]["num_workers"],
                                          drop_last=True)

        # Instantiate scheduler
        scheduler = lr_scheduler.OneCycleLR(optimizer,
                                            max_lr=self.config["training"]["learning_rate"],
                                            total_steps=len(train_loader) * self.config["training"]["epochs"])

        def save_model(step, max_checkpoints=5):
            os.makedirs(checkpoints_dir, exist_ok=True)

            checkpoints = glob.glob(checkpoints_dir + "/*.pth")
            checkpoints = sorted(checkpoints,
                                 key=os.path.getmtime,
                                 reverse=True)
            if len(checkpoints) >= max_checkpoints:
                for p in checkpoints[max_checkpoints-1:]:
                    print(
                        f"Removing checkpoint due to max checkpoint limit ({max_checkpoints}): {p}")
                    os.remove(p)

            state_dict = model.module.state_dict() if data_parallel else model.state_dict()
            torch.save(state_dict,
                       os.path.join(checkpoints_dir, 'model-{}.pth'.format(step)))

        # Train
        global_step = 0
        for epoch in range(self.config["training"]["epochs"]):
            model.train()

            epoch_loss = 0
            epoch_time = time.time()
            for _, (img, gt_score, gt_geo, ignored_map) in enumerate(train_loader):
                start_time = time.time()

                assert model.training, "Model is not in training mode!"

                img, gt_score, gt_geo, ignored_map = \
                    img.to(device), \
                    gt_score.to(device), \
                    gt_geo.to(device), \
                    ignored_map.to(device)
                pred_score, pred_geo = model(img)

                # print(f"pred_geo.shape: {pred_geo.shape}")

                # # XXX Rescale geometry values
                # input_h = img.shape[2]
                # input_w = img.shape[3]
                # # print(f"input_h: {input_h}")
                # # print(f"input_w: {input_w}")

                # pred_geo[:, [0, 1], :, :] *= input_h
                # pred_geo[:, [2, 3], :, :] *= input_w

                loss = criterion(gt_score, pred_score.to(gt_score.dtype),
                                 gt_geo, pred_geo.to(gt_geo.dtype),
                                 ignored_map)

                epoch_loss += loss.item()

                optimizer.zero_grad()
                if GOT_AMP and self.config["training"]["mixed_precision"]:
                    with amp.scale_loss(loss, optimizer) as scaled_loss:
                        scaled_loss.backward()
                else:
                    loss.backward()
                optimizer.step()
                scheduler.step()
                # train_loader.dataset.step()

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
                if global_step % self.config["training"]["checkpoint"]["frequency"] == 0:
                    # Testing

                    # Put model in eval mode
                    model.eval()

                    # No need gradient during testing
                    with torch.no_grad():
                        for _, (img, gt_score, gt_geo, ignored_map) in enumerate(test_loader):
                            img, gt_score, gt_geo, ignored_map = img.to(device), \
                                gt_score.to(device), \
                                gt_geo.to(device), \
                                ignored_map.to(device)

                            pred_score, pred_geo = model(img)

                            writer.add_images("image",
                                              img * torch.tensor(normalization_params["std"]).reshape(1, 3, 1, 1).to(
                                                  device) + torch.tensor(normalization_params["mean"]).reshape(1, 3, 1, 1).to(device),
                                              global_step=global_step)

                            writer.add_images("score_map/gt",
                                              gt_score,
                                              global_step=global_step)
                            writer.add_images("score_map/pred",
                                              pred_score,
                                              global_step=global_step)

                            for i in range(5):
                                if i == 4:
                                    writer.add_images(f"geo_map_{i}/gt",
                                                      (gt_geo[:, [i], :,
                                                              :] / math.pi) + 0.5,
                                                      global_step=global_step)
                                    writer.add_images(f"geo_map_{i}/pred",
                                                      (pred_geo[:, [
                                                       i], :, :] / math.pi) + 0.5,
                                                      global_step=global_step)
                                else:
                                    writer.add_images(f"geo_map_{i}/gt",
                                                      gt_geo[:, [i], :, :],
                                                      global_step=global_step)
                                    writer.add_images(f"geo_map_{i}/pred",
                                                      pred_geo[:, [i], :, :],
                                                      global_step=global_step)

                            # Break after first batch
                            break

                    # Put model back to training mode
                    model.train()

                    save_model(global_step,
                               self.config["training"]["checkpoint"]["max_checkpoints"])

            epoch_loss /= int(file_num/self.config["training"]["batch_size"])
            print('epoch: {:4d}, loss: {:.5f}, time {:.1f} s'.format(
                epoch+1,
                epoch_loss,
                time.time() - epoch_time
            ))
            writer.add_scalar("loss/train/epoch",
                              epoch_loss,
                              global_step=epoch)

        # Save final model and exit
        save_model(global_step,
                   self.config["training"]["checkpoint"]["max_checkpoints"])
        writer.close()

        print("Done!")


def parse_args():
    parser = argparse.ArgumentParser("EAST trainer")
    parser.add_argument("--config_path",
                        type=str,
                        default="configs/config.yaml",
                        help="Training config file path.")
    args = parser.parse_args()
    return args


def main(args):
    trainer = Trainer(config_path=args.config_path)
    trainer.train()


if __name__ == '__main__':
    # Import Apex AMP for mixed precision training
    try:
        from apex import amp
        GOT_AMP = True
        print("Apex AMP is available!")
    except ImportError:
        GOT_AMP = False
        print("Apex AMP is NOT available!")

    args = parse_args()
    main(args)
