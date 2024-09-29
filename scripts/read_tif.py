import tifffile as tiff
file_path = '/home/PanWei/Documents/Depth-Anything-V2/output/02/0000000071_pred.tif'
disaprity = tiff.imread(file_path)
print(disaprity[100][100])