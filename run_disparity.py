from os.path import abspath, dirname, join, isfile 
import sys
import os
if (os.system("cl.exe")):
    os.environ['PATH'] += ';'+r"C:\Program Files (x86)\Microsoft Visual Studio\2017\Community\VC\Tools\MSVC\14.16.27023\bin\Hostx64\x64"
if (os.system("cl.exe")):
    raise RuntimeError("cl.exe still not found, path probably incorrect")
sys.path.append(abspath(dirname(dirname(__file__))))



'''
from utils.video_utils import (load_video_from_file_name,
                               get_stereo_frame_from_video_capture,
                               split_stereo_frame_into_left_and_right_frames,
                               compute_background_mask)

from utils.py_cuda_interface import cuda_compute_disparity
'''
#########

#############
import numpy as np
import cv2
import math

def cuda_compute_disparity(image_right, image_left,
                           window_size, foreground_right,
                           foreground_left,
                           block_shape=(512, 1, 1),
                           grid_shape=(1, 1, 1)):
    assert image_left.shape == image_right.shape
    if image_left.dtype == np.uint8:
        print('converting Uint8 image to float32')
        image_left = image_left.astype(np.float32)
        image_right = image_right.astype(np.float32)
    import pycuda.autoinit
    import pycuda.driver as drv
    from pycuda.compiler import SourceModule
    cuda_filename = 'compute_disparity.cu'
    cuda_kernel_source = open(cuda_filename, 'r').read()
    cuda_module = SourceModule(cuda_kernel_source)
    compute_disparity = cuda_module.get_function('computeDisparity')

    img_height = image_left.shape[0]
    img_width = image_left.shape[1]

    calculated_disparity = np.zeros(shape=(img_height, img_width), dtype=np.float32)
    compute_disparity(
        drv.In(image_left),
        drv.In(image_right),
        np.int32(window_size),
        np.int32(img_height),
        np.int32(img_width),
        drv.In(foreground_right),
        drv.In(foreground_left),
        drv.Out(calculated_disparity),
        block=block_shape,
        grid=grid_shape
    )
    return calculated_disparity

def compute_background_mask(left_image, right_image):
    from cv2.bgsegm import createBackgroundSubtractorMOG
    from cv2 import dilate, erode, getStructuringElement
    bgSub = createBackgroundSubtractorMOG()
    bgSub.apply(left_image)
    bg_mask = bgSub.apply(right_image)
    # Dilate
    kernel = getStructuringElement(cv2.MORPH_RECT, (30, 30))
    dilated = dilate(bg_mask, kernel)
    return dilated


if __name__=='__main__':
    output_location = 'output.avi'
    cap1 = cv2.VideoCapture(r'..\videos\wheelhouse_bowfar1_cut.mp4')
    cap2 = cv2.VideoCapture(r'..\videos\wheelhouse_bowfar2_cut.mp4')
    #cap1 = cv2.VideoCapture(r'E:\Wheelhouse Bow Far\wheelhouse_bowfar1_night.avi')
    #cap2 = cv2.VideoCapture(r'E:\Wheelhouse Bow Far\wheelhouse_bowfar2_night.avi')
    cap1.grab()
    cap2.grab()
    print(cap1)
    print(cap2)
    
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    height = int(cap1.get(cv2.CAP_PROP_FRAME_HEIGHT))
    width = int(cap1.get(cv2.CAP_PROP_FRAME_WIDTH))
    fps = int(cap1.get(cv2.CAP_PROP_FPS))
    out = cv2.VideoWriter(output_location, fourcc,fps , (width,height))
    
    count = 0
    
    while(True):
        if count == 10:
            break
        count = count + 1
        #left_img, right_img = cv2.imread("./tsucuba_left.png"), cv2.imread("./tsucuba_right.png")
        f1,left_img = cap1.read()
        f2, right_img = cap2.read()
        #left_img = cv2.GaussianBlur(left_img,(5,5),10)
        #right_img = cv2.GaussianBlur(right_img,(5,5),10)
        #cv2.imshow('Image:', left_img)
        #cv2.waitKey(0)
        #print('Computing background mask...')
        bg_mask = compute_background_mask(left_img, right_img)
        #print('Computing disparity on GPU...')
        disparity_img = cuda_compute_disparity(
                image_left=left_img,
                image_right=right_img,
                foreground_left=np.ones(shape=(left_img.shape[0:1]),
                                        dtype=np.uint8),
                foreground_right=np.ones(shape=(left_img.shape[0:1]),
                                         dtype=np.uint8),
                window_size=20,
                block_shape=(512, 1, 1),
                grid_shape=(math.ceil(left_img.shape[0]* left_img.shape[1]/512), 1, 1)
            )
        test_point1 = 5*disparity_img[300][100]
        test_point2 = 5*disparity_img[1500][1000]
        print(test_point1,test_point2)
        d1 = 3.6*12.8016/test_point1
        d2 = 3.6*12.8016/test_point2
        colored_disparity = cv2.cvtColor(5*disparity_img,cv2.COLOR_GRAY2RGB).astype(np.uint8)
        print(np.amin(np.asarray(disparity_img)))
        #disparity_img.convertTo(image0, CV_32FC3, 1/255.0);
        #test_point1 = (np.asarray(disparity_img)/255)[300][100]
        #test_point2 = (np.asarray(disparity_img)/255)[1500][1000]
        
        
        print("Depth of point1: " + str(d1) + " metres")
        print("Depth of point2: " + str(d2) + " metres")
        
        
        #print('Got : ')
        #print(disparity_img)
        #print(np.amax(disparity_img))
        #cv2.imshow('Image:', left_img)
        #cv2.imshow('Disparity:', 5 * disparity_img.astype(np.uint8))
        print(count)
        #cv2.imwrite('haha1.jpg',5 * disparity_img.astype(np.uint8))
        
        #print(left_img.shape)
        #print(disparity_img.shape)
        out.write(cv2.flip(colored_disparity,180))   
        if cv2.waitKey(1) & 0xFF == ord('q'): #press q to quit
            break
    
    cap1.release()
    cap2.release()
    out.release()
    cv2.destroyAllWindows()