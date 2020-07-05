import argparse
import glob
import os

import lanms
import matplotlib.pyplot as plt
import numpy as np
import torch
import yaml
from PIL import Image, ImageDraw
from torchvision import transforms

from dataset import get_rotate_mat
from models import EAST


def resize_img(img):
    '''resize image to be divisible by 32
    '''

    w, h = img.size
    resize_w = w
    resize_h = h

    resize_h = resize_h if resize_h % 32 == 0 else int(resize_h / 32) * 32
    resize_w = resize_w if resize_w % 32 == 0 else int(resize_w / 32) * 32
    img = img.resize((resize_w, resize_h), Image.BILINEAR)
    ratio_h = resize_h / h
    ratio_w = resize_w / w

    return img, ratio_h, ratio_w


def load_pil(img, preprocessing_params):
    '''convert PIL Image to torch.Tensor
    '''

    t = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(**preprocessing_params)
    ])
    return t(img).unsqueeze(0)


def is_valid_poly(res, score_shape, scale):
    '''check if the poly in image scope
    Input:
            res        : restored poly in original image
            score_shape: score map shape
            scale      : feature map -> image
    Output:
            True if valid
    '''

    cnt = 0
    for i in range(res.shape[1]):
        if res[0, i] < 0 or res[0, i] >= score_shape[1] * scale or \
                res[1, i] < 0 or res[1, i] >= score_shape[0] * scale:
            cnt += 1
    return True if cnt <= 1 else False


def restore_polys(valid_pos, valid_geo, score_shape, scale=4):
    '''restore polys from feature maps in given positions
    Input:
            valid_pos  : potential text positions <numpy.ndarray, (n,2)>
            valid_geo  : geometry in valid_pos <numpy.ndarray, (5,n)>
            score_shape: shape of score map
            scale      : image / feature map
    Output:
            restored polys <numpy.ndarray, (n,8)>, index
    '''

    polys = []
    index = []
    # print(f"valid_pos: {valid_pos.dtype}")
    valid_pos *= scale
    d = valid_geo[:4, :]  # 4 x N
    angle = valid_geo[4, :]  # N,

    for i in range(valid_pos.shape[0]):
        x = valid_pos[i, 0]
        y = valid_pos[i, 1]
        y_min = y - d[0, i]
        y_max = y + d[1, i]
        x_min = x - d[2, i]
        x_max = x + d[3, i]
        rotate_mat = get_rotate_mat(-angle[i])

        temp_x = np.array([[x_min, x_max, x_max, x_min]]) - x
        temp_y = np.array([[y_min, y_min, y_max, y_max]]) - y
        coordidates = np.concatenate((temp_x, temp_y), axis=0)
        res = np.dot(rotate_mat, coordidates)
        res[0, :] += x
        res[1, :] += y

        if is_valid_poly(res, score_shape, scale):
            index.append(i)
            polys.append([res[0, 0], res[1, 0], res[0, 1], res[1, 1],
                          res[0, 2], res[1, 2], res[0, 3], res[1, 3]])
    return np.array(polys), index


def get_boxes(score, geo, score_thresh=0.9, nms_thresh=0.2, scale=4):
    '''get boxes from feature map
    Input:
            score       : score map from model <numpy.ndarray, (1,row,col)>
            geo         : geo map from model <numpy.ndarray, (5,row,col)>
            score_thresh: threshold to segment score map
            nms_thresh  : threshold in nms
    Output:
            boxes       : final polys <numpy.ndarray, (n,9)>
    '''

    score = score[0, :, :]
    xy_text = np.argwhere(score > score_thresh)  # n x 2, format is [r, c]
    if xy_text.size == 0:
        return None

    xy_text = xy_text[np.argsort(xy_text[:, 0])]
    valid_pos = xy_text[:, ::-1].copy()  # n x 2, [x, y]
    valid_geo = geo[:, xy_text[:, 0], xy_text[:, 1]]  # 5 x n
    polys_restored, index = restore_polys(
        valid_pos, valid_geo, score.shape, scale=scale)
    if polys_restored.size == 0:
        return None

    boxes = np.zeros((polys_restored.shape[0], 9), dtype=np.float32)
    boxes[:, :8] = polys_restored
    boxes[:, 8] = score[xy_text[index, 0], xy_text[index, 1]]
    boxes = lanms.merge_quadrangle_n9(boxes.astype('float32'), nms_thresh)
    return boxes


def adjust_ratio(boxes, ratio_w, ratio_h):
    '''refine boxes
    Input:
            boxes  : detected polys <numpy.ndarray, (n,9)>
            ratio_w: ratio of width
            ratio_h: ratio of height
    Output:
            refined boxes
    '''

    if boxes is None or boxes.size == 0:
        return None

    boxes[:, [0, 2, 4, 6]] /= ratio_w
    boxes[:, [1, 3, 5, 7]] /= ratio_h
    return np.around(boxes)


def detect(img, model, device,
           scale,
           preprocessing_params,
           score_thresh=0.9,
           nms_thresh=0.2):
    '''detect text regions of img using model
    Input:
            img   : PIL Image
            model : detection model
            device: gpu if gpu is available
            scale : image / feature map
            score_thresh: threshold to segment score map
            nms_thresh  : threshold in nms
    Output:
            detected polys
    '''

    img, ratio_h, ratio_w = resize_img(img)

    with torch.no_grad():
        score, geo = model(load_pil(img,
                                    preprocessing_params).to(device))

    boxes = get_boxes(score.squeeze(0).cpu().numpy(),
                      geo.squeeze(0).cpu().numpy(),
                      score_thresh=score_thresh,
                      nms_thresh=nms_thresh,
                      scale=scale)

    return adjust_ratio(boxes, ratio_w, ratio_h)


def plot_boxes(img, boxes):
    '''plot boxes on image
    '''

    if boxes is None:
        return img

    draw = ImageDraw.Draw(img)
    for box in boxes:
        draw.polygon([
            (box[0], box[1]),
            (box[2], box[3]),
            (box[4], box[5]),
            (box[6], box[7])
        ],
            outline=(0, 255, 0)  # GX: green color outline
        )
    return img


def detect_dataset(model, device, test_img_path, submit_path):
    '''detection on whole dataset, save .txt results in submit_path
    Input:
            model        : detection model
            device       : gpu if gpu is available
            test_img_path: dataset path
            submit_path  : submit result for evaluation
    '''
    img_files = os.listdir(test_img_path)
    img_files = sorted([os.path.join(test_img_path, img_file)
                        for img_file in img_files])

    for i, img_file in enumerate(img_files):
        print('evaluating {} image'.format(i), end='\r')
        boxes = detect(Image.open(img_file), model, device)

        seq = []
        if boxes is not None:
            seq.extend([','.join([str(int(b))
                                  for b in box[:-1]]) + '\n' for box in boxes])

        with open(os.path.join(submit_path, 'res_' + os.path.basename(img_file).replace('.jpg', '.txt')), 'w') as f:
            f.writelines(seq)


class Predictor(object):
    def __init__(self, config_path, device=None):
        self.config_path = config_path
        self._parse_config()

        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = device

        self._load_model()
        self._get_scope()
        self._compute_scale()
        self._get_preprocessing_params()

        print(f"scope: {self.scope}")
        print(f"scale: {self.scale}")

    def _parse_config(self):
        with open(self.config_path, "r") as f:
            self.config = yaml.load(f, Loader=yaml.Loader)

    def _load_model(self, checkpoint_path=None):
        # Instantiate model from configuration file
        model = EAST.from_config_file(self.config_path)

        if checkpoint_path is None or os.path.isdir(checkpoint_path):
            checkpoint_dir = os.path.join(self.config["training"]["prefix"], "checkpoints") \
                if checkpoint_path is None else checkpoint_path
            checkpoints = glob.glob(checkpoint_dir + "/*.pth")
            checkpoints = sorted(checkpoints,
                                 key=os.path.getmtime,
                                 reverse=True)
            if len(checkpoints) > 0:
                checkpoint_path = checkpoints[0]
            else:
                print(f"Warning, no checkpoint found in {checkpoint_dir}")
        elif not os.path.isfile(checkpoint_path):
            raise FileNotFoundError(
                f"Checkpoint path not found: {checkpoint_path}")
        model.load_state_dict(torch.load(checkpoint_path,
                                         map_location=torch.device("cpu")))

        # Move model to the right computation device
        model.to(self.device)

        # Put model in evaluation mode
        model.eval()

        self.model = model

    def _get_scope(self):
        """Input image size."""
        self.scope = self.config["model"]["scope"]

    def _compute_scale(self):
        """Compute scale."""
        dummy_out, _ = self.model(torch.rand(1, 3, self.scope, self.scope,
                                             device=self.device))
        scale = dummy_out.size(2)/self.scope
        # XXX This is inverse of what used in training for restoring polygons
        self.scale = int(1/scale)

    def _get_preprocessing_params(self):
        self.preprocessing_params = self.model.get_preprocessing_params()

    def predict(self, img_path,
                save_img=True, out_img_path="./prediction.jpg",
                return_img=False,
                score_thresh=0.9,
                nms_thresh=0.2):
        img = Image.open(img_path)
        boxes = detect(img,
                       model=self.model,
                       device=self.device,
                       scale=self.scale,
                       preprocessing_params=self.preprocessing_params,
                       score_thresh=score_thresh,
                       nms_thresh=nms_thresh)
        if save_img:
            plot_img = plot_boxes(img, boxes)
            os.makedirs(os.path.dirname(out_img_path), exist_ok=True)
            plot_img.save(out_img_path)

        if return_img:
            return boxes, plot_img
        else:
            return boxes

        print("Done!")

    def predict_dir(self, input_dir, output_dir="./predictions/"):
        """Predict using images in a directory.

        Args:
            input_dir ([type]): Input directory containing images to be predicted.
            output_dir ([type]): Output directory to store predicted results.
        """
        images = glob.glob(f"{input_dir}/**/*.jpg", recursive=True)

        print(f"Found {len(images)} images in {input_dir}")

        os.makedirs(os.path.join(output_dir, "img"), exist_ok=True)
        os.makedirs(os.path.join(output_dir, "pred"), exist_ok=True)
        for p in images:
            boxes = self.predict(p,
                                 save_img=True,
                                 out_img_path=os.path.join(output_dir, "img", os.path.basename(p)))
            if boxes is not None:
                with open(os.path.join(output_dir, "pred", os.path.splitext(os.path.basename(p))[0] + ".txt"), "w") as f:
                    for b in boxes:
                        f.write(
                            f"{','.join([str(int(v)) for v in b.tolist()])}\n")

        print("Done!")


def parse_args():
    parser = argparse.ArgumentParser("EAST trainer")
    parser.add_argument("--config_path",
                        type=str,
                        default="configs/config.yaml",
                        help="Training config file path.")
    parser.add_argument("--input",
                        type=str,
                        help="Input image to be predicted.")
    parser.add_argument("--input_dir",
                        type=str,
                        help="Input directory containing images to be predicted.")
    parser.add_argument("--output_dir",
                        type=str,
                        default="./predictions/",
                        help="Output directory to store predicted results.")
    args = parser.parse_args()
    return args


def main(args):
    predictor = Predictor(config_path=args.config_path)
    if args.input_dir and os.path.isdir(args.input_dir):
        predictor.predict_dir(args.input_dir, output_dir=args.output_dir)
    elif args.input and os.path.isfile(args.input):
        boxes = predictor.predict(args.input,
                                  save_img=True,
                                  out_img_path=os.path.join(args.output_dir, os.path.basename(args.input)))
        if boxes is not None:
            with open(os.path.join(args.output_dir, os.path.splitext(os.path.basename(args.input))[0] + ".txt"), "w") as f:
                for b in boxes:
                    f.write(f"{','.join([str(int(v)) for v in b.tolist()])}\n")
    elif args.input_dir:
        raise FileNotFoundError(f"Input directory not found: {args.input_dir}")
    elif args.input:
        raise FileNotFoundError(f"Input file not found: {args.input}")


if __name__ == '__main__':
    args = parse_args()
    main(args)
