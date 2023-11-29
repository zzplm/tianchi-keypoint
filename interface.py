import hashlib
import os.path

import pandas as pd
import numpy as np
import json
basepath = r"C:\Users\28144\Desktop\TianChi-FashionAi-keypoints\fashionAi_Data\test"
model_save_path = "C://Users//28144//Desktop//gitcode//ViTPose-Implementation-main//model"


keypoint_mapping = {
    'neckline_left': 0,
    'neckline_right': 1,
    'center_front': 2,
    'shoulder_left': 3,
    'shoulder_right': 4,
    'armpit_left': 5,
    'armpit_right': 6,
    'waistline_left': 7,
    'waistline_right': 8,
    'cuff_left_in': 9,
    'cuff_left_out': 10,
    'cuff_right_in': 11,
    'cuff_right_out': 12,
    'top_hem_left': 13,
    'top_hem_right': 14,
    'waistband_left': 15,
    'waistband_right': 16,
    'hemline_left': 17,
    'hemline_right': 18,
    'crotch': 19,
    'bottom_left_in': 20,
    'bottom_left_out': 21,
    'bottom_right_in': 22,
    'bottom_right_out': 23
}
def readImagePicture():
    df = pd.read_csv(basepath+"//test.csv")
    for key in keypoint_mapping.keys():
        # 为DataFrame添加新列，初始值为"-1_-1_-1"
        df[key] = ['0_0_0'] * len(df)
    new_file_path = 'new_data.csv'
    df.to_csv(new_file_path, index=False)


if __name__ =="__main__":
    listP = readImagePicture()




