import cv2
import numpy as np
class ReadFile:
    def __init__(self):
        super(ReadFile,self).__init__()


    # 读取图像
    def readFile(self,filename):
        self.img_src= cv2.imdecode(np.fromfile(filename, dtype=np.uint8), 1)
        # self.img_src = cv2.imread(filename)
        h,w,c = self.img_src.shape
        return h,w,c,self.img_src

