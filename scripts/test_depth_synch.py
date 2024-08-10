import argparse
import torch
import matplotlib
import cv2
import time
import sys
import numpy as np

sys.path.append('/home/ares/Desktop/weison/Depth-Anything-V2')

from depth_anything_v2.dpt import DepthAnythingV2

model_configs = {
    'vits': {'encoder': 'vits', 'features': 64, 'out_channels': [48, 96, 192, 384]},
    'vitb': {'encoder': 'vitb', 'features': 128, 'out_channels': [96, 192, 384, 768]},
    'vitl': {'encoder': 'vitl', 'features': 256, 'out_channels': [256, 512, 1024, 1024]},
    'vitg': {'encoder': 'vitg', 'features': 384, 'out_channels': [1536, 1536, 1536, 1536]}
}
 
def get_depth_path(old_name):  
    dirname_parts = old_name.split('/')  
    dirname_parts[-2] = 'groundtruth_depth' 

    filename_parts = dirname_parts[-1].split('_')  
    filename_parts[-4] = 'groundtruth_depth'

    new_filename = '_'.join(filename_parts)

    dirname_parts[-1] = new_filename

    new_dirname = '/'.join(dirname_parts)

    print(f"depth image : {new_dirname}")
    # exit()
    return new_dirname
    


def predict_depth(raw_image, encoder, input_size=518):

    depth_anything = DepthAnythingV2(**model_configs[encoder])
    depth_anything.load_state_dict(torch.load(f'checkpoints/depth_anything_v2_{args.encoder}.pth', map_location='cpu'))
    depth_anything = depth_anything.to(DEVICE).eval()


    
    start_time = time.time()  
    
    depth = depth_anything.infer_image(raw_image, input_size)
    
    end_time =  time.time()
    print(f"inferce time: {(end_time - start_time)*1000} ms")

    return depth

def normalize_depth(depth, grayscale=False):
    cmap = matplotlib.colormaps.get_cmap('Spectral_r')

    depth = (depth - depth.min()) / (depth.max() - depth.min()) * 255.0
    depth = depth.astype(np.uint8)
        
    if grayscale:
        depth = np.repeat(depth[..., np.newaxis], 3, axis=-1)
    else:
        depth = (cmap(depth)[:, :, :3] * 255)[:, :, ::-1].astype(np.uint8)

    return depth
  
def get_ratio(pred_depth,ground_truth):

    gray_ground_truth = ground_truth[..., 0]
    non_zero_indices = np.nonzero(gray_ground_truth)
    
    if len(non_zero_indices[0]) == 0:
        # No non-zero values found
        return None

    # Get the position of the first non-zero pixel
    fir_nonzero_pos = (non_zero_indices[0][0], non_zero_indices[1][0])

    print(ground_truth[fir_nonzero_pos[0]][fir_nonzero_pos[1]])
    print(pred_depth[fir_nonzero_pos[0]][fir_nonzero_pos[1]])

    exit()



    return None




if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Test depth synchronization between ground truth and predicted depth maps by depth anything.')

    # uncessary arguments
    parser.add_argument('--input-size', type=int, default=518)


    parser.add_argument('--raw_img_path', type=str, required=True, help='Path of ground truth depth maps.')
    parser.add_argument('--encoder', type=str, default='vitl', choices=['vits', 'vitb', 'vitl', 'vitg'])
    args = parser.parse_args()

    DEVICE= 'cuda' if torch.cuda.is_available() else 'cpu'

    raw_image = cv2.imread(args.raw_img_path)

    pred_depth_raw = predict_depth(raw_image, args.encoder)
    pred_depth = normalize_depth(pred_depth_raw)

    ground_truth = cv2.imread(get_depth_path(args.raw_img_path))

    split_region = np.ones((50, raw_image.shape[1], 3), dtype=np.uint8) * 255


    ratio = get_ratio(pred_depth_raw,ground_truth)

    

    combined_image = cv2.vconcat([raw_image, split_region,pred_depth,split_region, ground_truth])
    
    while True:
        cv2.imshow('combined', combined_image)

        key = cv2.waitKey(1)

        if key == ord('q'):
            print("Q key pressed, exiting...")
            break
    # closing all open windows
    cv2.destroyAllWindows()






    













    






