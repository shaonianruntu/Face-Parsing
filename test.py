import argparse
import os
import os.path as osp

import torch
import torchvision.transforms as transforms

from bisenetv2 import BiSeNetV2

import numpy as np
from scipy.io import savemat
import matplotlib.pyplot as plt
import seaborn as sns

import cv2
from PIL import Image

sns.set()

os.environ["CUDA_VISIBLE_DEVICES"] = "3"
"""
0: background 背景
1: skin 大脸瓜子
2: r_brow + l_brow 左右眉毛
3: r_eye + l_eye 左右眼睛
4: eye_g 眼镜
5: r_ear + l_ear 耳朵
6: ear_r 耳环
7: nose 鼻子
8: mouth 口腔
9: u_lip 上嘴唇
10: l_lip 下嘴唇
11: neck 脖子
12: neck_l 项链
13: cloth 衣服
14: hair 头发
15: hat 帽子
"""
ATTRIBUTE_NAME = [
    'background', 'skin', 'brow', 'eye', 'eye_g', 'ear', 'ear_r', 'nose',
    'mouth', 'u_lip', 'l_lip', 'neck', 'neck_l', 'cloth', 'hair', 'hat'
]

to_tensor = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
])


def oneshot(out_shot, img, atts_to_use, path, basename):
    parsing_save_path = osp.join(path, "parsing")
    joint_save_path = osp.join(path, "joint")
    if not os.path.isdir(parsing_save_path):
        os.makedirs(parsing_save_path)
    if not os.path.isdir(joint_save_path):
        os.makedirs(joint_save_path)

    temp1 = out_shot * 255
    for i in atts_to_use:
        name, _ = os.path.splitext(basename)
        mask_basename = name + "_" + str(i) + "_" + ATTRIBUTE_NAME[i] + ".png"
        temp = Image.fromarray(temp1[:, :, i]).convert('L')
        temp.save(osp.join(parsing_save_path, mask_basename))

        joint = np.array(img) * (out_shot[:, :, i])[:, :, np.newaxis]
        cv2.imwrite(osp.join(joint_save_path, mask_basename),
                    cv2.cvtColor(joint, cv2.COLOR_RGB2BGR))


def heatmap(feature_map, atts_to_use, path, basename):
    new_path = osp.join(path, "heatmap")
    if not os.path.isdir(new_path):
        os.makedirs(new_path)
    for i in atts_to_use:
        f, ax1 = plt.subplots()
        basename, _ = os.path.splitext(basename)
        mask_basename = basename + "_" + str(
            i) + "_" + ATTRIBUTE_NAME[i] + ".png"
        sns.heatmap(feature_map[i],
                    vmin=0,
                    vmax=1,
                    cmap='coolwarm',
                    center=0.5)
        plt.axis('off')
        f.savefig(osp.join(new_path, mask_basename))
        plt.close()


class FaceSeg(object):
    def __init__(self,
                 n_classes,
                 atts_to_use,
                 pretrain_model_path,
                 mouth_joint=False,
                 gpu=0):
        self.n_classes = n_classes
        self.atts_to_use = atts_to_use
        self.pre_train_model = pretrain_model_path
        self.mouth_joint = mouth_joint
        self.device = torch.device('cuda', gpu)
        self.net = BiSeNetV2(self.n_classes).to(self.device)
        self.net.load_state_dict(torch.load(self.pre_train_model))
        self.net.eval()

    def feed_data(self, data):
        self.var_L = data.to(self.device)
        with torch.no_grad():
            self.fake_H = self.net(self.var_L)
            return self.fake_H

    def generateBisenetV2(self, img):
        w, h = img.size
        image1 = img.resize((512, 512), Image.BILINEAR)
        image = torch.unsqueeze(to_tensor(image1), 0)
        result = self.feed_data(image)
        # heat = result.squeeze(0).cpu().numpy()
        # heatmap(heat, self.atts_to_use, parsing_path, ele)
        out_sig = torch.sigmoid(result)
        # saving seg results of test images
        out_shot = out_sig.squeeze(0).cpu().numpy()  # numpy (16*512*512)
        # oneshot(out_shot, self.atts_to_use, parsing_path, ele)

        ##########################################
        out_shot = np.transpose(out_shot, axes=[1, 2, 0])  # numpy(512*512*16)
        # out_shot = np.reshape(out_shot,(h,w,16))
        out_shot = cv2.resize(out_shot, (w, h),
                              interpolation=cv2.INTER_NEAREST)
        ##########################################################
        # do dilation transformation to eye
        dilate_kernel = np.ones((3, 3), np.uint8)
        out_shot[:, :, 3] = cv2.dilate(out_shot[:, :, 3], dilate_kernel)
        if self.mouth_joint:
            # put the lips and mouth together to get the final mouths
            out_shot[:, :,
                     8] = out_shot[:, :, 8] + out_shot[:, :,
                                                       9] + out_shot[:, :, 10]
        # binarize all results
        out_shot[out_shot < 0.3] = 0  # 0.5
        out_shot[out_shot >= 0.3] = 1
        return out_shot

    def get_parsing_index(self, out_shot, sketch):
        out_shot[out_shot < 0.5] = 0
        out_shot[out_shot >= 0.5] = 1
        # 16*512*512
        parsing_result = np.zeros(out_shot[0].shape)
        for i in range(len(out_shot)):
            if i != 4:  # The problem of eye_g 眼镜 is not considered here
                index = np.where(out_shot[i] == 1)
                parsing_result[index[0], index[1]] = i

        # only erode hair
        parsing_result = self.Erode_Hair_up(parsing_result, sketch)

        # handle forehead
        face_parsing = out_shot[1]
        temps = self.GetPartParsing(parsing_result, [1, 2, 3, 4, 7, 8, 9, 10])
        face = self.mix(temps)
        forehead = np.bitwise_xor(face_parsing.astype(np.int64),
                                  face.astype(np.int64))
        index = np.where(forehead == 1)
        parsing_result[index[0], index[1]] = 1

        return parsing_result

    def Erode_Hair_up(self, parsing, sketch):
        temp_hair = np.zeros(parsing.shape) + 1
        if 14 in np.unique(parsing):  # have hair
            index = np.where(parsing == 14)
            temp_hair[index[0], index[1]] = 0
            temp_hair = cv2.erode(temp_hair, np.ones((3, 3)), iterations=6)

        temp_bg = np.zeros(parsing.shape) + 1
        if 0 in np.unique(parsing):  # have bg
            index = np.where(parsing == 0)
            temp_bg[index[0], index[1]] = 0
            temp_bg = cv2.erode(temp_bg, np.ones((3, 3)), iterations=6)

        temp_ear = np.zeros(parsing.shape)
        if 5 in np.unique(parsing):  # 5： r_ear + l_ear 耳朵
            index = np.where(parsing == 5)
            temp_ear[index[0], index[1]] = 5

        index = np.where((temp_bg == 0) & (temp_hair == 0) & (sketch == 0))
        parsing[index[0], index[1]] = 0
        index = np.where((temp_bg == 0) & (temp_hair == 0) & (sketch == 0)
                         & (temp_ear == 5))
        parsing[index[0], index[1]] = 5
        return parsing

    def mix(self, faces):
        result = np.zeros((faces[0].shape[0], faces[0].shape[1]))
        for face in faces:
            index = np.where(face == 1)
            result[index[0], index[1]] = 1
        return result

    def GetPartParsing(self, parsing, flags):
        flag_parsings = []
        for flag in flags:
            flag_parsing = np.zeros((parsing.shape[0], parsing.shape[1]))
            index = np.where(parsing == flag)
            flag_parsing[index[0], index[1]] = 1
            flag_parsings.append(flag_parsing)
        return flag_parsings

    def parsing_colorful(self, parsing, colors):
        vis_parsing_anno_color = np.zeros(
            (parsing.shape[0], parsing.shape[1], 3)) + 245
        for i in range(0, 16):
            index = np.where(parsing == i)
            vis_parsing_anno_color[index[0], index[1], :] = colors[0][i]
        return vis_parsing_anno_color


def is_image_file(filename):
    IMG_EXTENSIONS = [
        '.jpg', '.jpeg', '.png', '.ppm', '.bmp', '.pgm', '.tif', '.tiff',
        'webp'
    ]
    filename_lower = filename.lower()
    return any(filename_lower.endswith(ext) for ext in IMG_EXTENSIONS)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--photo_path',
                        type=str,
                        default='/data/fangnan/photostick/CUHK4',
                        help="test image dataset path")
    parser.add_argument(
        '--photo_name',
        type=str,
        default='photos',
        help=
        "test image dataset name. Need to be combined with `photo_path`, or you can use `photo_path` directly."
    )
    parser.add_argument('--save_path',
                        type=str,
                        default='./result',
                        help="dataset_path")

    parser.add_argument('--pretrain_model',
                        type=str,
                        default='./pre_model/149999.pth',
                        help="pre-trained model path.")
    parser.add_argument('--n_classes',
                        type=int,
                        default=16,
                        help="number of segmentation class.")
    parser.add_argument('--atts_to_use',
                        type=int,
                        default=[0, 1, 2, 3, 7, 8],
                        help="number of segmentation class.")
    parser.add_argument(
        '--mouth_joint',
        type=bool,
        default=True,
        help=
        "'True' means put the lips and mouth together to get the final mouth.")

    parser.add_argument('--gpu', type=int, default=0, help='GPU Number')
    opts = parser.parse_args()

    opts.photo_path = osp.join(opts.photo_path, opts.photo_name)
    opts.save_path = osp.join(
        opts.save_path,
        opts.photo_name,
    )

    if not os.path.exists(opts.save_path):
        os.makedirs(opts.save_path)

    mat_save_path = osp.join(opts.save_path, 'mat')
    if not os.path.exists(mat_save_path):
        os.makedirs(mat_save_path)

    face_seg = FaceSeg(opts.n_classes, opts.atts_to_use, opts.pretrain_model,
                       opts.mouth_joint, opts.gpu)

    for name in sorted(os.listdir(opts.photo_path)):
        faces_path = os.path.join(opts.photo_path, name)
        if not is_image_file(faces_path):
            continue

        img = Image.open(faces_path).convert("RGB")
        out_shot = face_seg.generateBisenetV2(img)

        oneshot(out_shot, np.array(img), opts.atts_to_use, opts.save_path,
                name)
        savemat(osp.join(mat_save_path,
                         name.split('.')[0] + ".mat"), {'parsing': out_shot})

        print(name)
