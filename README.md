### 比较 DepAny2 的 inference 结果和 groudtruth
默认使用 vits 模型
```
python3 ./scripts/test_depth_synch.py --raw_img_path depth_selection/val_selection_cropped/image/2011_09_26_drive_0002_sync_image_0000000005_image_02.png
```

<div style="display: flex; justify-content: space-between; align-items: center;">
  <div style="text-align: center; width: 45%;">
    <img src="assets/weison/02.png" alt="Image 1" style="width: 100%;"/>
    <p><strong>grayscale映射rgb模式</strong></p>
  </div>
  <div style="text-align: center; width: 45%;">
    <img src="assets/weison/03.png" alt="Image 2" style="width: 100%;"/>
    <p><strong>grayscale模式</strong></p>
  </div>
</div>

### 基本的恒定的比值

0.012
$$
\frac{1/\text{model ouput} }{\text{ground truth}}
$$
### 误差值计算
kitti使用 SLLog ( Scale Invariant Logarithmic Eorror)
$$
\begin{aligned}D(y,y^{*})&=\quad\frac1{2n^2}\sum_{i,j}\left((\log y_i-\log y_j)-(\log y_i^*-\log y_j^*)\right)^2\\&=\quad\frac1n\sum_id_i^2-\frac1{n^2}\sum_{i,j}d_id_j\quad=\quad\frac1n\sum_id_i^2-\frac1{n^2}\left(\sum_id_i\right)^2\end{aligned}
$$
关于为何是 scale invariant 的， 参考论文[link](https://proceedings.neurips.cc/paper_files/paper/2014/file/7bccfde7714a1ebadf06c5f4cea752c1-Paper.pdf)
### 模型
1.2G的最大参数量的 checkpoint model在2060上跑kitti depth dataset 显存不足
放在3080Ti上能够正常跑

### 在 kitti depth dataset 1216 x 352 的图片中随机选取一张测试运行时长
<div style="text-align:center;">
 <img src="assets/weison/01.png" width="400" height="250">
</div>


## 目录结构

```
├── assets
│   ├── examples
│   └── examples_video
├── checkpoints
├── depth_anything_v2
│   ├── dinov2_layers
│   │   └── __pycache__
│   ├── __pycache__
│   └── util
│       └── __pycache__
├── depth_selection
│   ├── myimg
│   ├── test_depth_completion_anonymous
│   │   ├── image
│   │   ├── intrinsics
│   │   └── velodyne_raw
│   ├── test_depth_prediction_anonymous
│   │   ├── image
│   │   └── intrinsics
│   └── val_selection_cropped
│       ├── groundtruth_depth
│       ├── image
│       ├── intrinsics
│       └── velodyne_raw
├── devkit
│   ├── cpp
│   ├── matlab
│   └── python
├── metric_depth
│   ├── assets
│   ├── dataset
│   │   └── splits
│   │       ├── hypersim
│   │       ├── kitti
│   │       └── vkitti2
│   ├── depth_anything_v2
│   │   ├── dinov2_layers
│   │   └── util
│   └── util
├── scripts
└── vis_depth
```


### 跑官方demo
```
python3 run.py --img-path depth_selection/val_selection_cropped/image/2011_09_26_drive_0002_sync_image_0000000008_image_03.png --pred-only 
```

### TODO
- 优化多图片共同显示
- 整理代码，提高复用性