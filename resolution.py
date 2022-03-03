# This script reads jpg files from Images directory a computes experimental resolution of each image.
# Resolution values are printing in terminal.

from pathlib import Path
from fourier_tool import fourier_tool as ft

folder_dir = 'Images'
images = Path(folder_dir).glob('*.jpg')
res=[[],[]]
for sample in images:
    res.append([str(sample)[7:-4], ft.fourier_inspect(sample)])
    print('Resolution in '+res[-1][0]+' is '+str(res[-1][1])+' nanometers.')
