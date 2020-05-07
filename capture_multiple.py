import gphoto2 as gp
import os 
import time
import cv2
import numpy as np

context = gp.Context()
camera = gp.Camera()
camera.init(context)
files=[]
path='{}/{}'.format(os.getcwd(), time.strftime('%S_%M_%H_%d_%m_%Y',time.localtime()))
if not os.path.exists(path):
    os.mkdir(path)
os.chdir(path)

for i in range(20):
    print('Capturing image')
    file_path = camera.capture(gp.GP_CAPTURE_IMAGE, context)
    camera.wait_for_event(gp.GP_EVENT_CAPTURE_COMPLETE, context)
    print('Camera file path: {0}/{1}'.format(file_path.folder, file_path.name))
    target = os.path.join(os.getcwd(), file_path.name)
    files.append(target)
    print('Copying image to', target)
    camera_file = gp.check_result(camera.file_get(
        file_path.folder, file_path.name, gp.GP_FILE_TYPE_NORMAL, context))
    #file_data = gp.check_result(gp.gp_file_get_data_and_size(camera_file))
    
    camera_file.save(target)

    #img = cv2.imread(target)
    #cv2.imshow('Captured_Image', img)
    #cv2.waitKey(500)
camera.exit(context)
print("Context cleaaared")
