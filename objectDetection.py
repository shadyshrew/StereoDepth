from imageai.Detection import VideoObjectDetection
import os
import tensorflow as tf
import matplotlib.pyplot as plt

execution_path = os.getcwd()

color_index = {'bus': 'red', 'handbag': 'steelblue', 'giraffe': 'orange', 'spoon': 'gray', 'cup': 'yellow', 'chair': 'green', 'elephant': 'pink', 'truck': 'indigo', 'motorcycle': 'azure', 'refrigerator': 'gold', 'keyboard': 'violet', 'cow': 'magenta', 'mouse': 'crimson', 'sports ball': 'raspberry', 'horse': 'maroon', 'cat': 'orchid', 'boat': 'slateblue', 'hot dog': 'navy', 'apple': 'cobalt', 'parking meter': 'aliceblue', 'sandwich': 'skyblue', 'skis': 'deepskyblue', 'microwave': 'peacock', 'knife': 'cadetblue', 'baseball bat': 'cyan', 'oven': 'lightcyan', 'carrot': 'coldgrey', 'scissors': 'seagreen', 'sheep': 'deepgreen', 'toothbrush': 'cobaltgreen', 'fire hydrant': 'limegreen', 'remote': 'forestgreen', 'bicycle': 'olivedrab', 'toilet': 'ivory', 'tv': 'khaki', 'skateboard': 'palegoldenrod', 'train': 'cornsilk', 'zebra': 'wheat', 'tie': 'burlywood', 'orange': 'melon', 'bird': 'bisque', 'dining table': 'chocolate', 'hair drier': 'sandybrown', 'cell phone': 'sienna', 'sink': 'coral', 'bench': 'salmon', 'bottle': 'brown', 'car': 'silver', 'bowl': 'maroon', 'tennis racket': 'palevilotered', 'airplane': 'lavenderblush', 'pizza': 'hotpink', 'umbrella': 'deeppink', 'bear': 'plum', 'fork': 'purple', 'laptop': 'indigo', 'vase': 'mediumpurple', 'baseball glove': 'slateblue', 'traffic light': 'mediumblue', 'bed': 'navy', 'broccoli': 'royalblue', 'backpack': 'slategray', 'snowboard': 'skyblue', 'kite': 'cadetblue', 'teddy bear': 'peacock', 'clock': 'lightcyan', 'wine glass': 'teal', 'frisbee': 'aquamarine', 'donut': 'mincream', 'suitcase': 'seagreen', 'dog': 'springgreen', 'banana': 'emeraldgreen', 'person': 'honeydew', 'surfboard': 'palegreen', 'cake': 'sapgreen', 'book': 'lawngreen', 'potted plant': 'greenyellow', 'toaster': 'ivory', 'stop sign': 'beige', 'couch': 'khaki'}

plt.rcParams['interactive'] == True
#plt.switch_backend('QT4Agg')
plt.switch_backend('TkAgg')
#plt.switch_backend('wxAgg')
figManager = plt.get_current_fig_manager()
#https://stackoverflow.com/questions/12439588/how-to-maximize-a-plt-show-window-using-python
#figManager.window.showMaximized() #QT4Agg
#figManager.frame.Maximize(True)   #wxAgg
figManager.window.state('zoomed') #TkAgg

plt.figure(1)

'''
def forFrame(frame_number, output_array, output_count):
    print("FOR FRAME " , frame_number)
    print("Output for each object : ", output_array)
    print("Output count for unique objects : ", output_count)
    print("------------END OF A FRAME --------------")
'''
    
def forFrame(frame_number, output_array, output_count, returned_frame):
    global plt
    plt.clf()

    this_colors = []
    labels = []
    sizes = []

    counter = 0
    
    for eachItem in output_count:
        counter += 1
        labels.append(eachItem + " = " + str(output_count[eachItem]))
        sizes.append(output_count[eachItem])
        this_colors.append(color_index[eachItem])
    
    plt.axis("off")
    plt.imshow(returned_frame, interpolation="none")
    plt.show()
    plt.pause(0.001)

detector = VideoObjectDetection()
detector.setModelTypeAsYOLOv3()
detector.setModelPath("..\models\yolo.h5")
detector.loadModel()
detector.detectObjectsFromVideo(
        input_file_path=os.path.join(execution_path, "..\videos\wheelhouse_bowfar1_night.avi"), 
        output_file_path=os.path.join(execution_path, "boats_detected_out_night") ,  
        frames_per_second=30,
        per_frame_function=forFrame, 
        minimum_percentage_probability=20, 
        return_detected_frame=True, 
        log_progress=True)
