import argparse
import torch
import matplotlib
import cv2
import time
import sys
import numpy as np
import os
import random
from PIL import Image


# for import depth_anything module
sys.path.append('./')



from depth_anything_v2.dpt import DepthAnythingV2

def align_depth_least_square(
    gt_arr: np.ndarray,
    pred_arr: np.ndarray,
    valid_mask_arr: np.ndarray,
    return_scale_shift=False,
    max_resolution=None,
):
    ori_shape = pred_arr.shape  # input shape

    gt = gt_arr.squeeze()  # [H, W]
    pred = pred_arr.squeeze()
    valid_mask = valid_mask_arr.squeeze()

    # Downsample
    if max_resolution is not None:
        scale_factor = np.min(max_resolution / np.array(ori_shape[-2:]))
        if scale_factor < 1:
            downscaler = torch.nn.Upsample(scale_factor=scale_factor, mode="nearest")
            gt = downscaler(torch.as_tensor(gt).unsqueeze(0)).numpy()
            pred = downscaler(torch.as_tensor(pred).unsqueeze(0)).numpy()
            valid_mask = (
                downscaler(torch.as_tensor(valid_mask).unsqueeze(0).float())
                .bool()
                .numpy()
            )

    assert (
        gt.shape == pred.shape == valid_mask.shape
    ), f"{gt.shape}, {pred.shape}, {valid_mask.shape}"

    gt_masked = gt[valid_mask].reshape((-1, 1))
    pred_masked = pred[valid_mask].reshape((-1, 1))

    # numpy solver
    _ones = np.ones_like(pred_masked)
    A = np.concatenate([pred_masked, _ones], axis=-1)
    X = np.linalg.lstsq(A, gt_masked, rcond=None)[0]
    scale, shift = X

    aligned_pred = pred_arr * scale + shift

    # restore dimensions
    aligned_pred = aligned_pred.reshape(ori_shape)

    print(f"scale is {scale}\n")
    print(f"shift is {shift}\n")
    if return_scale_shift:
        return aligned_pred, scale, shift
    else:
        return aligned_pred

model_configs = {
    'vits': {'encoder': 'vits', 'features': 64, 'out_channels': [48, 96, 192, 384]},
    'vitb': {'encoder': 'vitb', 'features': 128, 'out_channels': [96, 192, 384, 768]},
    'vitl': {'encoder': 'vitl', 'features': 256, 'out_channels': [256, 512, 1024, 1024]},
    'vitg': {'encoder': 'vitg', 'features': 384, 'out_channels': [1536, 1536, 1536, 1536]}
}

# using in original dataset
def get_depth_path(old_name):  

    # get the corresponse path of a rgb_image in the folder structure of kitti depth dataset

    dirname_parts = old_name.split('/')  
    dirname_parts[-2] = 'groundtruth_depth' 

    filename_parts = dirname_parts[-1].split('_')  
    filename_parts[-4] = 'groundtruth_depth'

    new_filename = '_'.join(filename_parts)

    dirname_parts[-1] = new_filename

    new_dirname = '/'.join(dirname_parts)

    return new_dirname
    
# for img and gt in the same dir
def get_depth_path_v2(old_name):
    dirname_parts = old_name.split('.')  
    dirname_parts[-2] = dirname_parts[-2] + "_gt" 
    new_dirname = '.'.join(dirname_parts)
    return new_dirname

def predict_depth(raw_image, encoder, input_size=518):
    r"""Return a single channel Dep2Anything predition."""

    depth_anything = DepthAnythingV2(**model_configs[encoder])
    depth_anything.load_state_dict(torch.load(f'checkpoints/depth_anything_v2_{args.encoder}.pth', map_location='cpu',weights_only=True))
    depth_anything = depth_anything.to(DEVICE).eval()
    
    start_time = time.time()  
    
    depth = depth_anything.infer_image(raw_image, input_size)
    
    end_time =  time.time()
    print(f"inferce time: {(end_time - start_time)*1000} ms")

    return depth


 
def inverse_depth(depth):
    depth = np.where(depth != 0, 1.0 / depth, 0)
    return depth

def depth_read(filename):
    # loads depth map D from png file
    # and returns it as a numpy array,
    # for details see readme.txt

    depth_png = np.array(Image.open(filename), dtype=int)
    # make sure we have a proper 16bit depth map here.. not 8bit!
    assert(np.max(depth_png) > 255)

    depth = depth_png.astype(np.float32) / 256.
    depth[depth_png == 0] = -1.
    return depth

def eval_depth(pred, target):
    assert pred.shape == target.shape

    thresh = torch.max((target / pred), (pred / target))

    d1 = torch.sum(thresh < 1.25).float() / len(thresh)
    d2 = torch.sum(thresh < 1.25 ** 2).float() / len(thresh)
    d3 = torch.sum(thresh < 1.25 ** 3).float() / len(thresh)

    diff = pred - target
    diff_log = torch.log(pred) - torch.log(target)

    abs_rel = torch.mean(torch.abs(diff) / target)
    sq_rel = torch.mean(torch.pow(diff, 2) / target)

    rmse = torch.sqrt(torch.mean(torch.pow(diff, 2)))
    rmse_log = torch.sqrt(torch.mean(torch.pow(diff_log , 2)))

    log10 = torch.mean(torch.abs(torch.log10(pred) - torch.log10(target)))
    silog = torch.sqrt(torch.pow(diff_log, 2).mean() - 0.5 * torch.pow(diff_log.mean(), 2))

    return {'d1': d1.item(), 'd2': d2.item(), 'd3': d3.item(), 'abs_rel': abs_rel.item(), 'sq_rel': sq_rel.item(), 
            'rmse': rmse.item(), 'rmse_log': rmse_log.item(), 'log10':log10.item(), 'silog':silog.item()}

def save_tif(img,prefix):
    Image.fromarray(img).save(f'./output/02/{prefix}_pred.tif')
    


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Test depth synchronization between ground truth and predicted depth maps by depth anything.')

    # uncessary arguments
    parser.add_argument('--input-size', type=int, default=518)


    # parser.add_argument('--raw_img_path', type=str, required=True, help='Path of ground truth depth maps.')
    parser.add_argument('--encoder', type=str, default='vitb', choices=['vits', 'vitb', 'vitl', 'vitg'])
    parser.add_argument('--grayscale',dest='grayscale', action='store_true', help='do not apply colorful palette')
    parser.add_argument('--sample-size',type=int,default=100)
    parser.add_argument('--img_dir',type=str)
    args = parser.parse_args()

    DEVICE= 'cuda' if torch.cuda.is_available() else 'cpu'


    # img_dir = '/home/ares/Desktop/weison/Depth-Anything-V2/data/test/'
    img_dir = args.img_dir
    # img_dir = 'depth_selection/val_selection_cropped/same_scene/'
    # all_images = [img_dir+f for f in os.listdir(img_dir) if f.endswith('.png') and f.startswith('2011_09_26_drive_0095')] 
    all_images = [img_dir+f for f in os.listdir(img_dir)]

    # print(all_images[0])

    all_images = [img for img in all_images if img.endswith('.png') and not img.endswith('_gt.png')]

    print(f'total {len(all_images)} images in this scene')



    for img_path in all_images:
        raw_image = cv2.imread(img_path)
        print("====================================")
        print(f'processing {img_path}')
        disparity_raw = predict_depth(raw_image, args.encoder)   
        # prefix = '.'.join((img_path.split("/")[-1]).split(".")[:-1])
        # save_tif(disparity_raw,prefix)
       
        print(disparity_raw[100][100])
        pred_raw = inverse_depth(disparity_raw)   
        print(pred_raw[100][100])
        gt_raw = depth_read(get_depth_path_v2(img_path)) 

        valid_mask = (gt_raw >= 0.001) & (gt_raw <= 20)
        # gt_raw 的均值大概是 10~11 左右，pred_raw 的均值在0.0045左右, 10.5/0.0045 = 2333.3333333333335,sacle 大约都是这么多
        # print(np.mean(pred_raw[valid_mask]))
        # continue
        pred = align_depth_least_square(gt_raw,pred_raw,valid_mask)

        pred_raw_ts = torch.from_numpy(pred)
        gt_raw_ts = torch.from_numpy(gt_raw)
        eval_results = eval_depth(pred_raw_ts[valid_mask], gt_raw_ts[valid_mask])
        print(eval_results)
        print("====================================")
