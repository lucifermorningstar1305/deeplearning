import cv2
from tqdm.auto import tqdm
import pandas as pd

class Resizer:
    def __init__(self):
        pass

    def image_resize(self, df, size=28):
        resize = size
        resized = {}

        for i in tqdm(range(df.shape[0])):
            image = df.loc[df.index[i]].values.reshape(137,236)
            _, thresh = cv2.threshold(image, 30, 255, cv2.THRESH_BINARY_INV+cv2.THRESH_OTSU)
            contours, _ = cv2.findContours(thresh, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)[-2:]
            idx = 0
            ls_xmin = []
            ls_xmax = []
            ls_ymin = []
            ls_ymax = []
            for cnt in contours:
                idx += 1
                x,y,w,h = cv2.boundingRect(cnt)
                ls_xmin.append(x)
                ls_ymin.append(y)
                ls_xmax.append(x+w)
                ls_ymax.append(y+h)
            xmin = min(ls_xmin)
            xmax = max(ls_xmax)
            ymin = min(ls_ymin)
            ymax = max(ls_ymax)
            roi = image[ymin:ymax, xmin:xmax]
            resized_roi = cv2.resize(roi, (resize, resize), interpolation=cv2.INTER_AREA)
            resized[df.index[i]] = resized_roi.reshape(-1)
        resized = pd.DataFrame(resized).T
        return resized