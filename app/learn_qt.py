import sys
import cv2
from PyQt5.QtWidgets import QDialog, QApplication,QMainWindow
from PyQt5.QtGui import QImage,QPixmap
from PyQt5.QtCore import QThread,pyqtSignal,pyqtSlot,QRect,QCoreApplication
from ui.dish import Ui_MainWindow
from image_segmentation import Segmentation
import time
import logging
logging.basicConfig(filename='logger.log',level=logging.INFO)

class Thread(QThread):
    changePixmap = pyqtSignal(QImage)
    def initApp(self,app):
        self.app = app

    def price(self,colors):
        '''
        price compute
        :return:
        '''
        price = 0
        price_dict = {'green':2,'orange':3,'yellow':5}
        for i in colors:
            price+=price_dict[i]

        return price

    def adjust_scale(self,rgbImage):
        # update width and height by image
        width, height = self.app.ui.frame.width(), self.app.ui.frame.height()
        coefficient_width = width / rgbImage.shape[1] / rgbImage.shape[0]
        coefficient_height = height / rgbImage.shape[0]
        if coefficient_width > coefficient_height:
            width = height * (rgbImage.shape[1] / rgbImage.shape[0])
        else:
            height = width * (rgbImage.shape[0] / rgbImage.shape[1])
        return width,height

    def run(self):
        cap = cv2.VideoCapture(0)
        # 0 init; 1 start; 2 end
        flag = 0
        # times
        times = 0
        # record
        last_result = []
        seg = Segmentation()
        seg.load_model()
        while True:
            ret, frame = cap.read()

            if ret:

                seg = Segmentation()
                seg.add_img(frame)
                seg.hough_circles()

                if seg.check_circles():
                    seg.get_circle_pixel(10)
                    result = list(seg.get_colors())
                    result.sort()
                    self.app.ui.pushButton.setText(QCoreApplication.translate(u"recogination", ', '.join(result)))

                    if times >=5:
                        self.app.ui.pushButton_2.setText(QCoreApplication.translate(u"recogination", '检测完成，请移走餐盘'))
                        flag = 2
                    else:
                        self.app.ui.pushButton_2.setText(QCoreApplication.translate(u"recogination", '检测中...'))
                        flag = 1

                    self.app.ui.price.display(self.price(result))
                    seg.draw_circles()
                    if result == last_result:
                        times += 1
                    else:
                        times = 0
                    last_result = result
                else:
                    self.app.ui.pushButton.setText(QCoreApplication.translate(u"recogination", 'prepare for detecting'))
                    self.app.ui.pushButton_2.setText(QCoreApplication.translate(u"recogination", '请放入餐盘...'))
                    self.app.ui.price.display(0)
                    times = 0
                    last_result = []
                    if flag == 2:
                        '''jie suan'''

                rgbImage = cv2.cvtColor(seg.img_rgb, cv2.COLOR_BGR2RGB)
                convertToQtFormat = QImage(rgbImage.data, rgbImage.shape[1], rgbImage.shape[0], QImage.Format_RGB888)

                width, height = self.adjust_scale(rgbImage)
                p = convertToQtFormat.scaled(width, height)
                self.app.label.setGeometry(QRect(0, 0, self.app.ui.frame.width(), self.app.ui.frame.height()))
                self.changePixmap.emit(p)
                time.sleep(0.1)

class AppWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.ui = Ui_MainWindow()
        self.ui.setupUi(self)

        self.label = self.ui.label_3
        th = Thread(self)
        th.initApp(self)
        th.changePixmap.connect(self.setImage)
        th.start()


    @pyqtSlot(QImage)
    def setImage(self, image):
        self.label.setPixmap(QPixmap.fromImage(image))

def run():
    app = QApplication(sys.argv)
    w = AppWindow()
    w.show()
    sys.exit(app.exec_())

if __name__ == '__main__':
    run()
