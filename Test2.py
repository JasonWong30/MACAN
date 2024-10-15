# coding:utf-8
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
import argparse
import torch
from torch.utils.data import DataLoader
from util.loader import Med_testdataset
from tqdm import tqdm
import torch.utils.data
import torch.nn.functional
import time
import numpy as np
from util.YUVandRGB import RGB2YCrCb
import torchvision.transforms as transforms
from PIL import Image
from util.convnet_utils import switch_deploy_flag, switch_conv_bn_impl
from Networks.net2 import MODEL as net

# To run, set the fused_dir, and the val path in the TaskFusionDataset.py    ./model/Fusion/fusionmodel_final.pth  ./model/Fusion/fusion_model.pth
def main(args):
    transform_test = transforms.Compose([
        transforms.ToTensor(),
    ])

    device = torch.device("cuda:{}".format(args.gpu) if torch.cuda.is_available() else "cpu")

    model = net(deploy=True)
    model.eval()
    model.to(device)
    # checkpoint = torch.load(args.model_test_path, map_location='cuda')
    # model.load_state_dict(checkpoint['model'], strict=True)
    model.load_state_dict(torch.load(args.model_test_path, map_location='cuda'))

    print('fusionmodel load done!')
    flag = True
    # test_dataset = Med_testdataset(args.data_dir, 'test_CT-MRI', 'test', 'CT', transform=transform_test)
    test_dataset = Med_testdataset(args.data_dir, 'test_PET-MRI', 'test', 'PET', transform=transform_test)
    # test_dataset = Med_testdataset(args.data_dir, 'test_IR-VI', 'test', 'PET', transform=transform_test)
    # test_dataset = Med_testdataset(args.data_dir, 'add_test_SPECT-MRI', 'test', 'SPECT', transform=transform_test)
    test_loader = DataLoader(
        dataset=test_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=False,
    )
    time_list = []
    test_loader.n_iter = len(test_loader)
    test_bar = tqdm(test_loader)
    with torch.no_grad():
        for it, (vis, ir, name) in enumerate(test_bar):
            start = time.time()
            vis = vis.to(device)
            ir = ir.to(device)
            # b,c,h,w = vis.shape

            if flag:
                vi_Y, vi_Cb, vi_Cr = RGB2YCrCb(vis)
                vi_Y = vi_Y.to(device)
                vi_Cb = vi_Cb.to(device)
                vi_Cr = vi_Cr.to(device)
            else:
                vi_Y = vis

            # vi_Y = torch.reshape(vi_Y,(b,c,768, 1024))
            # ir = torch.reshape(ir, (b,c,768, 1024))
            fused_img = model(vi_Y.float(), ir.float())

            # fused_img = torch.reshape(fused_img, (h, w))
            # fused_img = YCbCr2RGB(fused_img, vi_Cb, vi_Cr)

            ones = torch.ones_like(fused_img)
            zeros = torch.zeros_like(fused_img)
            fused_img = torch.where(fused_img > ones, ones, fused_img)
            fused_img = torch.where(fused_img < zeros, zeros, fused_img)
            fused_image = fused_img.cpu().numpy()
            fused_image = fused_image.transpose((0, 2, 3, 1))
            fused_image = (fused_image - np.min(fused_image)) / (
                    np.max(fused_image) - np.min(fused_image)
            )
            fused_image = np.uint8(255.0 * fused_image)
            end = time.time()
            time_list.append(end - start)
            for k in range(len(name)):
                image = fused_image[k, :, :, :]
                image = Image.fromarray(image[:,:,0])
                save_path = os.path.join(args.save_dir, name[k])
                image.save(save_path)
                # cv2.imwrite( save_path, image[:,:,0])
    print(time_list)

def convert(args):
    device = torch.device("cuda:{}".format(args.gpu) if torch.cuda.is_available() else "cpu")
    switch_conv_bn_impl('DBB')
    switch_deploy_flag(False)
    model = net(deploy=False)
    model.eval()
    model.to(device)
    checkpoint = torch.load(args.model_path, map_location='cuda')
    model.load_state_dict(checkpoint['model'], strict=True)
    for m in model.modules():
        if hasattr(m, 'switch_to_deploy'):
            m.switch_to_deploy()
    torch.save(model.state_dict(), args.model_test_path)
    print('1')

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Run MACAN with pytorch')
    ## model
    parser.add_argument('--model_path', type=str, default='./output_dir/checkpoint.pth') #IVIF-20?
    parser.add_argument('--model_test_path', type=str, default='./test_checkpoint/test_checkpoint.pth')
    parser.add_argument('--data_dir', type=str, default='./data/test')
    parser.add_argument('--save_dir', '-save_dir', type=str, default='./MCAFusion/')
    parser.add_argument('--batch_size', '-B', type=int, default=1)
    parser.add_argument('--gpu', '-G', type=int, default=0)
    parser.add_argument('--Train', type=bool, default=False)
    parser.add_argument('--num_workers', '-j', type=int, default=8)
    args = parser.parse_args()
    os.makedirs(args.save_dir, exist_ok=True)
    print('| testing %s on GPU #%d with pytorch' % ('SeAFusion', args.gpu))
    convert(args)
    main(args)


