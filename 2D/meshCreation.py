import numpy as np
from init import defineConstants as init 
import sys,time
from PIL import Image

def default():
    print(f'No file found continuing with default case.')
    #mesh defination
    x=np.arange(init.N_POINTS_X)
    y=np.arange(init.N_POINTS_Y)
    X,Y=np.meshgrid(x,y,indexing="ij")
    #masking the mesh to detect the obstacle
    obj_mask=(np.sqrt((X-init.OBJ_CENTER_X)**2 +(Y-init.OBJ_CENTER_Y)**2)<init.OBJ_RADII_IDX)
    return [X,Y],obj_mask


def defineObstacle(obstaclePath):
    
    if not obstaclePath:
      return default()

    try:
        print(f'Attempting to read \"{obstaclePath}\".')
        img= Image.open(obstaclePath)
        img=img.convert('L')
        object=np.array(img)
        time.sleep(3)
    
    except FileNotFoundError:
        print(f'Cannot read file at \"{obstaclePath}\".')
        time.sleep(2)
        print(f'System will now exit.')
        time.sleep(1)
        sys.exit()
    
    shapex=np.arange(object.shape[0])
    shapey=np.arange(object.shape[1])

    X,Y=np.meshgrid(shapex,shapey,indexing='ij')

    object=object/np.max(object)       
    obj_mask=object[X,Y] < .2

    return [X,Y],obj_mask


