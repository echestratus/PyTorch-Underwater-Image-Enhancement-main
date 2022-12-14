'''
Author:Xuelei Chen(chenxuelei@hotmail.com)
Usgae:
python test.py --checkpoint CHECKPOINTS_PATH
'''
import os
import torch
import numpy as np
from PIL import Image
from model import PhysicalNN
import argparse
from torchvision import transforms
import datetime
import math
import cv2


def main(checkpoint):

    # ori_dirs = []
    # for image in os.listdir(imgs_path):
    #     ori_dirs.append(os.path.join(imgs_path, image))

    # Check for GPU

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Load model
    model = PhysicalNN()
    model = torch.nn.DataParallel(model).to(device)
    print("=> loading trained model")
    checkpoint = torch.load(checkpoint, map_location=device)
    model.load_state_dict(checkpoint['state_dict'])
    print("=> loaded model at epoch {}".format(checkpoint['epoch']))
    model = model.module
    model.eval()

    testtransform = transforms.Compose([
                transforms.ToTensor(),
            ])
    unloader = transforms.ToPILImage()

    cap = cv2.VideoCapture(0)
    while(True):
        ret, frame = cap.read()

        inp = testtransform(frame).unsqueeze(0)
        inp = inp.to(device)
        out = model(inp)

        corrected = unloader(out.cpu().squeeze(0))
        frames = np.array(corrected)
        print(frames.shape)
        cv2.imshow('imageenhancement',frames)
        cv2.imshow('original',frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
    # for imgdir in ori_dirs:
    #     img_name = (imgdir.split('/')[-1]).split('.')[0]
    #     img = Image.open(imgdir)
    #     inp = testtransform(img).unsqueeze(0)
    #     inp = inp.to(device)
    #     out = model(inp)
    #
    #     corrected = unloader(out.cpu().squeeze(0))
    #     dir = '{}/results_{}'.format(result_path, checkpoint['epoch'])
    #     if not os.path.exists(dir):
    #         os.makedirs(dir)
    #     corrected.save(dir+'/{}corrected.png'.format(img_name))


if __name__ == '__main__':

    # parser = argparse.ArgumentParser()
    # parser.add_argument('--checkpoint', help='checkpoints path', required=True)
    # args = parser.parse_args()
    # checkpoint = args.checkpoint
    main(checkpoint=r'PyTorch-Underwater-Image-Enhancement-main/PyTorch_UnderwaterImageEnhancement/checkpoints/model_best_2842.pth.tar')
