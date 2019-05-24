'''
collect color data with label
通过摄像头捕捉物体的边缘颜色,记录颜色rgb的值,并且调整光线等外界因素,来收集不同的环境下的颜色的rgb值
对不同颜色的物体分别记录
rgb作为训练数据,color作为标签
'''
import numpy as np
import cv2
import time
import pandas as pd
import image_segmentation
import sqlalchemy

engine = sqlalchemy.create_engine('mysql+pymysql://kai:vsi666666@localhost/learn?charset=utf8')


# 需要调整制定颜色color
def color_analysis(seg):
    print('*'*40)
    color = 'orange'
    for c in seg.colors:
        df = pd.DataFrame(c,columns = ['r','g','b'])
        df['color'] = color
        df.to_sql('colors',engine,if_exists='append',index=False)

if __name__ == '__main__':
    cap_index = 0
    cap = cv2.VideoCapture(cap_index)

    while (True):
        # Capture frame-by-frame
        ret, frame = cap.read()
        seg = image_segmentation.Segmentation(frame)
        seg.hough_circles()
        for radius in range(3,20):
            seg.get_circle_pixel(radius)
            # color_analysis(seg)
        print(list(seg.get_colors()))
        seg.draw_circles()

        # Display the resulting frame
        cv2.imshow('frame', seg.img_rgb)
        # time.sleep(1)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # When everything done, release the capture
    cap.release()
    cv2.destroyAllWindows()
