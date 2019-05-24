import cv2
import math
import numpy as np
import pickle

model_path = 'model/di_model.pkl'

class Segmentation():
    def __init__(self):
        '''

        '''

    def add_img(self,mat):
        self.img_rgb = mat
        self.img = mat[:,:,0]

    def load_model(self):
        self.clf = pickle.load(open(model_path, 'rb'))

    def read_img(self,filepath):
        img = cv2.imread(filepath,0)
        img_rgb = cv2.imread(filepath)
        self.img = img
        self.img_rgb = img_rgb

    def process_img(self):
        scale = 5
        x = int(self.img.shape[1]/scale)
        y = int(self.img.shape[0]/scale)
        self.img = cv2.resize(self.img,(x,y))
        self.img_rgb = cv2.resize(self.img_rgb,(x,y))

    def candy(self):
        edges = cv2.Canny(self.img, 100, 200)
        self.edges = edges

    def hough_circles(self):
        img_min_edge = min(self.img.shape)
        circles = cv2.HoughCircles(self.img, cv2.HOUGH_GRADIENT, 1.8, int(img_min_edge/4), minRadius=int(img_min_edge/8), maxRadius=int(img_min_edge/3))

        self.circles = circles

    def check_circles(self):
        if self.circles is None:
            return False
        else:
            return True

    def nocircles(self):
        if self.circles is None:
            return True
        else:
            return False

    def get_rects_from_circles(self):
        if self.nocircles():
            return
        else:
            rects = []
            for i in self.circles[0,:]:
                x1 = int(i[0]-i[2])
                x2 = int(i[0]+i[2])
                y1 = int(i[1]-i[2])
                y2 = int(i[1]+i[2])
                rect = [x1,x2,y1,y2]
                rects.append(rect)
            self.rects = rects


    def draw_circles(self):
        '''
        画圆
        :return:
        '''
        if self.nocircles():
            return

        for i in self.circles[0,:]:
            cv2.circle(self.img_rgb,(i[0],i[1]),int(i[2]),[100,130,0],5)

    def get_circle_pixel(self,radius=0):
        '''
        得到圆环周围的像素
        :param radius:
        :return:
        '''
        angles = 360
        points = []
        colors = []

        if self.nocircles():
            self.points = np.array(points)
            self.colors = np.array(colors)
            return

        for i in self.circles[0,:]:
            p = []
            c = []
            r = i[2]-radius
            cx = i[0]
            cy = i[1]
            for angle in range(angles):
                x = cx-math.cos(angle)*r
                y = cy+math.sin(angle)*r
                try:
                    p.append(np.array([int(x),int(y)]))
                    c.append(self.img_rgb[int(y),int(x)])
                except:
                    continue

            points.append(np.array(p))
            colors.append(np.array(c))

        self.points = np.array(points)
        self.colors = np.array(colors)


    def color_judge(self,colors):
        '''

        :param colors:
        :return:
        '''
        color_dict = {0:'green',1:'orange',2:'yellow'}
        # LOAD MODEL
        pre = self.clf.predict(colors)
        pre_unique,count = np.unique(pre,return_counts= True)
        max_index = np.argmax(count)

        return color_dict[pre_unique[max_index]]

    def get_colors(self):
        for c in self.colors:
            yield self.color_judge(c)


    def imshow(self):
        cv2.imshow('img',self.img_rgb)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    def get_pic(self):
        for i in self.rects:
            print(i)
            cv2.imshow('img',self.img[i[0]:i[1],i[2]:i[3]])
            cv2.waitKey(0)
            cv2.destroyAllWindows()
        # cv2.destroyAllWindows()

    def imshow_edges(self):
        cv2.imshow('edges',self.edges)
        cv2.waitKey(0)
        cv2.destroyAllWindows()




