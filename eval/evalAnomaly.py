# Copyright (c) OpenMMLab. All rights reserved.
import os
import cv2
import glob
import torch
import random
import numpy as np
import os.path as osp
import torch.nn.functional as F
import torchvision.transforms as T

from PIL import Image
from erfnet import ERFNet
from argparse import ArgumentParser
from ood_metrics import fpr_at_95_tpr, calc_metrics, plot_roc, plot_pr,plot_barcode
from sklearn.metrics import roc_auc_score, roc_curve, auc, precision_recall_curve, average_precision_score

seed = 42

# general reproducibility
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)

NUM_CHANNELS = 3
NUM_CLASSES = 20
# gpu training specific
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = True

# Preprocessing
image_transform = T.Compose([
  T.Resize((512, 1024), Image.BILINEAR), T.ToTensor()
])

mask_transform = T.Compose([
  T.Resize((512, 1024), Image.NEAREST)
])

def main():
    parser = ArgumentParser()
    parser.add_argument(
        "--input",
        default="/home/shyam/Mask2Former/unk-eval/RoadObsticle21/images/*.webp",
        nargs="+",
        help="A list of space separated input images; "
        "or a single glob pattern such as 'directory/*.jpg'",
    )  
    parser.add_argument('--loadDir',default="../trained_models/")
    parser.add_argument('--loadWeights', default="erfnet_pretrained.pth")
    parser.add_argument('--loadModel', default="erfnet.py")
    parser.add_argument('--subset', default="val")  #can be val or train (must have labels)
    parser.add_argument('--datadir', default="/home/shyam/ViT-Adapter/segmentation/data/cityscapes/")
    parser.add_argument('--num-workers', type=int, default=4)
    parser.add_argument('--batch-size', type=int, default=1)
    parser.add_argument('--cpu', action='store_true')
    # new arguments
    parser.add_argument('--method', type=str, default='msp')
    parser.add_argument('--temperature', type=float, default=1.0)
    args = parser.parse_args()  # argparse.Namespace object that contained arguments
    anomaly_score_list = []
    ood_gts_list = []

    if not os.path.exists('results.txt'):
        open('results.txt', 'w').close()
    file = open('results.txt', 'a')

    modelpath = args.loadDir + args.loadModel
    weightspath = args.loadDir + args.loadWeights

    # print ("Loading model: " + modelpath)
    # print ("Loading weights: " + weightspath)

    model = ERFNet(NUM_CLASSES)

    if (not args.cpu):
        model = torch.nn.DataParallel(model).cuda()

    def load_my_state_dict(model, state_dict):  #custom function to load model when not all dict elements
        own_state = model.state_dict()
        for name, param in state_dict.items():
            if name not in own_state:
                if name.startswith("module."):
                    own_state[name.split("module.")[-1]].copy_(param)
                else:
                    print(name, " not loaded")
                    continue
            else:
                own_state[name].copy_(param)
        return model

    model = load_my_state_dict(model, torch.load(weightspath, map_location=lambda storage, loc: storage, weights_only=True))
    # print ("Model and weights LOADED successfully")
    model.eval()
    
    for path in glob.glob(os.path.expanduser(str(args.input[0]))):
        # images = torch.from_numpy(np.array(Image.open(path).convert('RGB'))).unsqueeze(0).float().cuda()
        # images = images.permute(0,3,1,2)
        images = image_transform(Image.open(path).convert('RGB')).unsqueeze(0).float().cuda()
        with torch.no_grad():
            result = model(images).squeeze(0)
        # print(result.shape)

        # methods
        if args.method == "void":
            anomaly_result = F.softmax(result, dim=0)[-1]
        else:
            result = result[:-1]  # remove background class
            if args.method == "msp":
                softmax_probs = F.softmax(result / args.temperature, dim=0)
                anomaly_result = 1.0 - torch.max(softmax_probs, dim=0)[0]
            elif args.method == "maxlogit":
                anomaly_result = -torch.max(result, dim=0)[0]
            elif args.method == "maxentropy":
                anomaly_result = torch.div(
                    torch.sum(-F.softmax(result, dim=0) * F.log_softmax(result, dim=0), dim=0),
                    torch.log(torch.tensor(result.size(0))),
                )

        anomaly_result = anomaly_result.data.cpu().numpy()
        # anomaly_result = 1.0 - np.max(result.squeeze(0).data.cpu().numpy(), axis=0)            
        pathGT = path.replace("images", "labels_masks")                
        if "RoadObsticle21" in pathGT:
           pathGT = pathGT.replace("webp", "png")
        if "fs_static" in pathGT:
           pathGT = pathGT.replace("jpg", "png")                
        if "RoadAnomaly" in pathGT:
           pathGT = pathGT.replace("jpg", "png")  

        mask = Image.open(pathGT)
        ood_gts = np.array(mask_transform(mask))

        if "RoadAnomaly" in pathGT:
            ood_gts = np.where((ood_gts==2), 1, ood_gts)
        if "LostAndFound" in pathGT:
            ood_gts = np.where((ood_gts==0), 255, ood_gts)
            ood_gts = np.where((ood_gts==1), 0, ood_gts)
            ood_gts = np.where((ood_gts>1)&(ood_gts<201), 1, ood_gts)

        if "Streethazard" in pathGT:
            ood_gts = np.where((ood_gts==14), 255, ood_gts)
            ood_gts = np.where((ood_gts<20), 0, ood_gts)
            ood_gts = np.where((ood_gts==255), 1, ood_gts)

        if 1 not in np.unique(ood_gts):
            continue              
        else:
             ood_gts_list.append(ood_gts)
             anomaly_score_list.append(anomaly_result)
        del result, anomaly_result, ood_gts, mask
        torch.cuda.empty_cache()

    file.write( "\n")

    ood_gts = np.array(ood_gts_list)
    anomaly_scores = np.array(anomaly_score_list)

    ood_mask = (ood_gts == 1)
    ind_mask = (ood_gts == 0)

    ood_out = anomaly_scores[ood_mask]
    ind_out = anomaly_scores[ind_mask]

    ood_label = np.ones(len(ood_out))
    ind_label = np.zeros(len(ind_out))
    
    val_out = np.concatenate((ind_out, ood_out))
    val_label = np.concatenate((ind_label, ood_label))

    prc_auc = average_precision_score(val_label, val_out)
    fpr = fpr_at_95_tpr(val_out, val_label)

    print(f'\t\tAUPRC score: {prc_auc*100.0:.3f}')
    print(f'\t\tFPR@TPR95: {fpr*100.0:.3f}')

    file.write(('    AUPRC score:' + str(prc_auc*100.0) + '   FPR@TPR95:' + str(fpr*100.0) ))
    file.close()

if __name__ == '__main__':
    main()