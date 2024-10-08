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

model_configs = {
    'vits': {'encoder': 'vits', 'features': 64, 'out_channels': [48, 96, 192, 384]},
    'vitb': {'encoder': 'vitb', 'features': 128, 'out_channels': [96, 192, 384, 768]},
    'vitl': {'encoder': 'vitl', 'features': 256, 'out_channels': [256, 512, 1024, 1024]},
    'vitg': {'encoder': 'vitg', 'features': 384, 'out_channels': [1536, 1536, 1536, 1536]}
}
 
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
    


def predict_depth(raw_image, encoder, input_size=518):
    r"""Return a single channel Dep2Anything predition."""

    depth_anything = DepthAnythingV2(**model_configs[encoder])
    depth_anything.load_state_dict(torch.load(f'checkpoints/depth_anything_v2_{args.encoder}.pth', map_location='cpu',weights_only=True))
    depth_anything = depth_anything.to(DEVICE).eval()


    
    start_time = time.time()  
    
    depth = depth_anything.infer_image(raw_image, input_size)
    
    end_time =  time.time()
    # print(f"inferce time: {(end_time - start_time)*1000} ms")

    return depth

def normalize_depth(depth):
    r'''normalize the single channel and return single channel '''


    depth = (depth - depth.min()) / (depth.max() - depth.min()) * 255.0
    depth = depth.astype(np.uint8)
        

    return depth

def get_groundtruth(raw_img_path,grayscale=False):
    r'''read the single channel depth image in kitti 
    and return a 3-channel image
    '''

    depth_image_path = get_depth_path(raw_img_path)
    ground_truth = one2three(cv2.imread(depth_image_path,0),grayscale)

    return ground_truth

def one2three(singchanimg,grayscale=False):
    r'''to turn a single channel depth image to 3 channel image based on whether to grayscale'''

    # print(f'singchanimg is {singchanimg}')
    if grayscale:
        # equal to directly return 3 times copy of single channel
        threechanimg = np.repeat(singchanimg[..., np.newaxis], 3, axis=-1).astype(np.uint8)

    else:
        cmap = matplotlib.colormaps.get_cmap('Spectral_r')
        threechanimg = (cmap(singchanimg)[:, :, :3] * 255)[:, :, ::-1].astype(np.uint8)
    
    return threechanimg

def normalize_gt(raw_img_path):
    depth_image_path = get_depth_path(raw_img_path)
    depth = cv2.imread(depth_image_path,0)

    min = np.min(depth[depth>0])
    max = np.max(depth[depth>0])
    depth[depth>0] = (depth[depth>0] - min) / (max - min) * 255.0
    depth = depth.astype(np.uint8)
    return depth

# 
def get_singchannel_ratio(pre,gt,sample_size):
    r''' input pre and gt single channel depth numpy image and return average ratio (original ratio calculation version)'''
    non_zero_indx = np.nonzero(gt)
    
    num_non_zero_index = len(non_zero_indx[0])
    ratios = []
    for i in range(sample_size):
        index = np.random.randint(num_non_zero_index)
        x = non_zero_indx[0][index]
        y = non_zero_indx[1][index]

        tmp_ratio = pre[x,y] / gt[x,y]  
        ratios.append(tmp_ratio)


    ratio_mean = np.mean(ratios)
    ratio_variance = np.var(ratios)
    # print(f"ratios is {ratios}")
    print(f"For sample size {sample_size}, ratio mean is {ratio_mean}, ratio variance is {ratio_variance}")

    return ratio_mean

def get_filter_ratio(pre,gt,sample_size):
    r'''given the valid masked and minmax-filtered image and calculate the ratio'''
    # print(f'the number of valid points{len(pre)}')
    ratios = []
    for i in range(sample_size):
        index = np.random.randint(len(pre))
        tmp_ratio = pre[index] / gt[index]
        ratios.append(tmp_ratio)

    ratio_mean = np.mean(ratios)
    ratio_variance = np.var(ratios)
    # print(f"ratios is {ratios}")
    # print(f"For sample size {sample_size}, mean is {ratio_mean}, variance is {ratio_variance}")

    return ratio_mean,ratio_variance
 



def get_ratio(pred_raw,raw_img_path,sample_size):
    depth_image_path = get_depth_path(raw_img_path)
    ground_truth = cv2.imread(depth_image_path,0)
    print(ground_truth)

    non_zero_indx = np.nonzero(ground_truth)
    
    if len(non_zero_indx[0]) == 0:
        return None

    num_non_zero_index = len(non_zero_indx[0])


    ratios = []

    # randomly pick sample_size non-zero index to calculate the ratio
    for i in range(sample_size):
        index = np.random.randint(num_non_zero_index)
        x = non_zero_indx[0][index]
        y = non_zero_indx[1][index]

        tmp_ratio = pred_raw[x,y] / ground_truth[x,y]  
        ratios.append(tmp_ratio)

    ratio_mean = np.mean(ratios)
    ratio_variance = np.var(ratios)

    print(f"the ratios is {ratios}")
    
    print(f"For sample size {sample_size}, mean is {ratio_mean}, variance is {ratio_variance}")

    return ratio_mean

def inverse_depth(depth):
    depth = np.where(depth != 0, 1.0 / depth, 0)
    return depth

def get_imageshow(a,b,c,d):
    r'''concat 4 images into a single image for display'''

    v_split_region = np.ones((50, a.shape[1], 3), dtype=np.uint8) * 255

    v_combined_image_1 = cv2.vconcat([a, v_split_region, b])
    v_combined_image_2 = cv2.vconcat([c,v_split_region,d])

    h_split_region = np.ones((v_combined_image_1.shape[0],50,3), dtype=np.uint8) * 255

    combined_image = cv2.hconcat([v_combined_image_1,h_split_region,v_combined_image_2])

    return combined_image

def limit_output(pred,min_depth_eval,max_depth_eval):
    pred[pred < min_depth_eval] = min_depth_eval
    pred[pred > max_depth_eval] = max_depth_eval
    pred[np.isinf(pred)] = max_depth_eval
    pred[np.isnan(pred)] = min_depth_eval

    return pred

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

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Test depth synchronization between ground truth and predicted depth maps by depth anything.')

    # uncessary arguments
    parser.add_argument('--input-size', type=int, default=518)


    # parser.add_argument('--raw_img_path', type=str, required=True, help='Path of ground truth depth maps.')
    parser.add_argument('--encoder', type=str, default='vitb', choices=['vits', 'vitb', 'vitl', 'vitg'])
    parser.add_argument('--grayscale',dest='grayscale', action='store_true', help='do not apply colorful palette')
    parser.add_argument('--sample-size',type=int,default=100)
    args = parser.parse_args()

    DEVICE= 'cuda' if torch.cuda.is_available() else 'cpu'


    # ground_truth = get_groundtruth(args.raw_img_path, args.grayscale)

    # pred_depth = normalize_depth(pred_raw)

    # groud_truth_norm = normalize_gt(args.raw_img_path)
    # # ratio = get_ratio(groud_truth_norm)
    # get_singchannel_ratio(pred_depth,groud_truth_norm,args.sample_size)

    # combined_image = get_imageshow(raw_image,pred_depth,ground_truth,pred_depth_scale)
    


    # single image test
    #  ----------------------------------------------- #
    #  没有归一化的 1/模型的输出 vs 没有归一化的 gt深度图
    # raw_image = cv2.imread(args.raw_img_path)
    # pred_raw = predict_depth(raw_image, args.encoder)
    # gt_raw = cv2.imread(get_depth_path(args.raw_img_path),0)
    # pred_inverse = inverse_depth(pred_raw)
    # get_singchannel_ratio(pred_inverse,gt_raw,args.sample_size)
    #  ----------------------------------------------- #

    # multiple image test
    # img_dir = 'depth_selection/val_selection_cropped/image/'
    img_dir = '/home/bwshen/Desktop/pw/Depth-Anything-V2/data/test/'
    # img_dir = 'depth_selection/val_selection_cropped/same_scene/'
    # all_images = [img_dir+f for f in os.listdir(img_dir) if f.endswith('.png') and f.startswith('2011_09_26_drive_0095')] 
    all_images = [img_dir+f for f in os.listdir(img_dir)]
    # This scenes change a lot: 2011_09_26_drive_0036
    # all_images = [img_dir+f for f in os.listdir(img_dir) if f.endswith('.png') and f.startswith('2011_09_26_drive_0036')] 

    # print(len(all_images))
    # exit()
    # all_images.sort()

    print(f'total {len(all_images)} images in this scene')
    # selected_images_paths = random.sample(all_images, 40)
    # 2011_09_26_drive_0095_sync_image_0000000245_image_03.png

    # all_images = [all_images[i] for i in range(0, len(all_images), 3)]

    # first test  
    # all_images = ['/home/bwshen/Desktop/pw/Depth-Anything-V2/depth_selection/val_selection_cropped/image/2011_09_26_drive_0002_sync_image_0000000005_image_02.png','/home/bwshen/Desktop/pw/Depth-Anything-V2/depth_selection/val_selection_cropped/image/2011_09_26_drive_0002_sync_image_0000000008_image_03.png','/home/bwshen/Desktop/pw/Depth-Anything-V2/depth_selection/val_selection_cropped/image/2011_09_26_drive_0002_sync_image_0000000014_image_03.png']

    # all_images_ = [
    #     '/home/bwshen/Desktop/pw/Depth-Anything-V2/depth_selection/val_selection_cropped/image/2011_09_26_drive_0023_sync_image_0000000077_image_03.png',
    #     '/home/bwshen/Desktop/pw/Depth-Anything-V2/depth_selection/val_selection_cropped/image/2011_09_26_drive_0023_sync_image_0000000080_image_02.png',
    #     '/home/bwshen/Desktop/pw/Depth-Anything-V2/depth_selection/val_selection_cropped/image/2011_09_26_drive_0023_sync_image_0000000083_image_03.png',
    #     '/home/bwshen/Desktop/pw/Depth-Anything-V2/depth_selection/val_selection_cropped/image/2011_09_26_drive_0023_sync_image_0000000086_image_02.png'
    # ]

    all_txt = [
        '/home/bwshen/Desktop/pw/Depth-Anything-V2/depth_selection/val_selection_cropped/intrinsics/2011_09_26_drive_0023_sync_image_0000000077_image_03.txt',
        '/home/bwshen/Desktop/pw/Depth-Anything-V2/depth_selection/val_selection_cropped/intrinsics/2011_09_26_drive_0023_sync_image_0000000080_image_02.txt',
        '/home/bwshen/Desktop/pw/Depth-Anything-V2/depth_selection/val_selection_cropped/intrinsics/2011_09_26_drive_0023_sync_image_0000000083_image_03.txt',
        '/home/bwshen/Desktop/pw/Depth-Anything-V2/depth_selection/val_selection_cropped/intrinsics/2011_09_26_drive_0023_sync_image_0000000086_image_02.txt'
    ]

    all_dep = [
        '/home/bwshen/Desktop/pw/Depth-Anything-V2/depth_selection/val_selection_cropped/groundtruth_depth/2011_09_26_drive_0023_sync_groundtruth_depth_0000000077_image_03.png',
        '/home/bwshen/Desktop/pw/Depth-Anything-V2/depth_selection/val_selection_cropped/groundtruth_depth/2011_09_26_drive_0023_sync_groundtruth_depth_0000000080_image_02.png',
        '/home/bwshen/Desktop/pw/Depth-Anything-V2/depth_selection/val_selection_cropped/groundtruth_depth/2011_09_26_drive_0023_sync_groundtruth_depth_0000000083_image_03.png',
        '/home/bwshen/Desktop/pw/Depth-Anything-V2/depth_selection/val_selection_cropped/groundtruth_depth/2011_09_26_drive_0023_sync_groundtruth_depth_0000000086_image_02.png'
    ]

    # all_images = [
    #     '/home/bwshen/Desktop/pw/Depth-Anything-V2/depth_selection/val_selection_cropped/image/2011_09_26_drive_0023_sync_image_0000000077_image_03.png',
    #     '/home/bwshen/Desktop/pw/Depth-Anything-V2/depth_selection/val_selection_cropped/image/2011_09_26_drive_0023_sync_image_0000000080_image_02.png',
    #     '/home/bwshen/Desktop/pw/Depth-Anything-V2/depth_selection/val_selection_cropped/image/2011_09_26_drive_0023_sync_image_0000000083_image_03.png',
    #     '/home/bwshen/Desktop/pw/Depth-Anything-V2/depth_selection/val_selection_cropped/image/2011_09_26_drive_0023_sync_image_0000000086_image_02.png'
    # ]
    # cout = 0 2011_09_26_drive_0023_sync_image_0000000086_image_02.png
    # crop image as Depth anything V1 
    for img_path in all_images:
        raw_image = cv2.imread(img_path)
        pred_raw = predict_depth(raw_image, args.encoder)
        gt_raw = cv2.imread(get_depth_path(img_path),0)  # orginal version, haven divided by 256.0
        gt_raw = depth_read(get_depth_path(img_path))  # new version, divided by 256.0
        pred_inverse= inverse_depth(pred_raw)


        # mask_ = pred_inverse < 1.2
        # pred_inverse = pred_inverse[mask]
        # gt_raw = gt_raw[mask]
        
        # only consider the depth < 10.0, which is significant for 3D reconstruction
        mask_ = gt_raw < 10.0


        # min max depth as depth anything v1
        min_depth_eval = 0.001
        # max_depth_eval = 80.0
        max_depth_eval = 10.0
        pred_inverse[pred_inverse<min_depth_eval] = min_depth_eval
        pred_inverse[pred_inverse>max_depth_eval] = max_depth_eval
        # cv2.imwrite(f'./output/{cout}_pred.png', pred_inverse.astype('float32')) #lost information after dot
        prefix = '.'.join((img_path.split("/")[-1]).split(".")[:-1])
        Image.fromarray(pred_inverse).save(f'./output/02/{prefix}_pred.tif')
        valid_mask = np.logical_and(gt_raw > min_depth_eval, gt_raw < max_depth_eval)

        # reload_img = np.array(Image.open(f'./output/{img_path.split("/")[-1]}_pred.tif'))

        # eigen_crop
        eigen_crop = True
        if eigen_crop:
            gt_height, gt_width = gt_raw.shape
            eval_mask = np.zeros(valid_mask.shape)
            eval_mask[int(0.3324324 * gt_height):int(0.91351351 * gt_height),
                          int(0.0359477 * gt_width):int(0.96405229 * gt_width)] = 1


        pred_inverse_vmask = pred_inverse[valid_mask]
        gt_raw_vmask = gt_raw[valid_mask]
        valid_mask = np.logical_and(valid_mask, eval_mask)
        valid_mask = np.logical_and(valid_mask, mask_)
        ratio_mean, ratio_var = get_filter_ratio(pred_inverse_vmask,gt_raw_vmask,args.sample_size)

        sav = pred_inverse_vmask.astype(np.uint8)
        
        pred_ratio = pred_inverse_vmask / ratio_mean

        # relative error
        error = np.abs(pred_ratio - gt_raw_vmask) / gt_raw_vmask
        error_mean = np.mean(error)
        error_variance = np.var(error)

        # mask = pred_ratio < 100
        # filtered_error = error[mask]
        # error_mean = np.mean(filtered_error)
        # variance = np.var(filtered_error)

        print(f'{ratio_mean} {ratio_var} {error_mean} {error_variance} ')

        if (error_variance>400):
            print("debug begin")
            index_of_max = np.argmax(error)
            print(f'最大{pred_ratio[index_of_max]}')
            print(f'最大{gt_raw_vmask[index_of_max]}')
            print(f'debug end')

        # cout += 1



        


    


    
    
    
    # #  ----------------------------------------------- #
    # # 归一化的  1/模型的输出 vs 归一化的  gt深度图
    # pred_depth_inverse_norm = normalize_depth(pred_inverse)
    # print(len(np.nonzero(pred_depth_inverse_norm)))
    # ground_truth_norm = normalize_depth(gt_raw)
    # get_singchannel_ratio(pred_depth_inverse_norm, ground_truth_norm ,args.sample_size)
    # #  ----------------------------------------------- #



    np.set_printoptions(threshold=np.inf)
    # print(pred_inverse)
    # # pred_inverse = np.clip(pred_inverse,0,255)
    # while True:
    #     cv2.imshow('h',pred_raw)
    #     # cv2.imshow('b',gt_raw)
    #     # cv2.imshow('v',pred_inverse)

    #     key = cv2.waitKey(1)

    #     if key == ord('q'):
    #         print("Q key pressed, exiting...")
    #         break
    cv2.destroyAllWindows()






    













    






