import hashlib
import os.path

import pandas as pd
import numpy as np
import json

from PIL import Image

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
reverse_mapping = {v: k for k, v in keypoint_mapping.items()}  # kv 翻转 都好找


def readCsvToJson():
    df = pd.read_csv(r'C:\Users\28144\Desktop\TianChi-FashionAi-keypoints\fashionAi_Data\test\new_data.csv')
    basepath = r"C:\Users\28144\Desktop\TianChi-FashionAi-keypoints\fashionAi_Data\test"
    savepath = r"C:\Users\28144\Desktop\TianChi-FashionAi-keypoints\fashionAi_Data\test\annotations"

    # df = pd.read_csv(r'C:\Users\28144\Desktop\TianChi-FashionAi-keypoints\fashionAi_Data\train\train.csv')
    # basepath = r"C:\Users\28144\Desktop\TianChi-FashionAi-keypoints\fashionAi_Data\train"
    # savepath = r"C:\Users\28144\Desktop\TianChi-FashionAi-keypoints\fashionAi_Data\train\annotations"

    blouseData = []
    blouseName = []
    outwearData = []
    outwearName = []
    dressData = []
    dressName = []
    skirtData = []
    skirtName = []
    trouserData = []
    trouserName = []
    for rows in df.iterrows():
        # 按行读取
        # 都要-1
        row = rows[1]

        # pathName = "name"
        pathVal = row[0].split("/")[2]
        # 类别
        # sortName = "categories"
        sortVal = row[1]
        if sortVal == 'blouse':
            ##0 1 2 3 4 5 6  0 0 9 10 11 12 13 14 15 0 0 0 0 0 0 0 0 0 直接-1 就不看了 遮挡的也无关紧要
            listBlouse = [0, 1, 2, 3, 4, 5, 6, 9, 10, 11, 12, 13, 14]
            blouseData.append(getKeyPoint(listBlouse, pathVal, row, sortVal))
            blouseName.append(pathVal.split(".")[0])
        elif sortVal == 'outwear':
            # 0 1 0 3 4 5 6 7 8 9 10 11 12 13 14 0 0 0 0 0 0 0 0 0
            listOutwear = [0, 1, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14]
            outwearData.append(getKeyPoint(listOutwear, pathVal, row, sortVal))
            outwearName.append(pathVal.split(".")[0])
        elif sortVal == 'dress':
            # 0 1 2 3 4 5 6 7 8 9 10 11 12 0 0 0 0 17 18
            listDress = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 17, 18]
            dressData.append(getKeyPoint(listDress, pathVal, row, sortVal))
            dressName.append(pathVal.split(".")[0])
        elif sortVal == 'skirt':
            ## 15 16 17 18
            listSkirt = [15, 16, 17, 18]
            skirtData.append(getKeyPoint(listSkirt, pathVal, row, sortVal))
            skirtName.append(pathVal.split(".")[0])
        elif sortVal == 'trousers':
            # 15 16 19 20 21 22 23
            listTrouser = [15, 16, 19, 20, 21, 22, 23]
            trouserData.append(getKeyPoint(listTrouser, pathVal, row, sortVal))
            trouserName.append(pathVal.split(".")[0])
        else:
            # 异常数据不管
            print("异常数据")

    ##data里面是数据 更改成coco
    cocoDataFBlouse = CocoAndSave(basepath, blouseData,blouseName)
    # 合并多个数据集字典为一个字典
    mergeAndSave(cocoDataFBlouse, savepath)
    cocoDataFOutwear= CocoAndSave(basepath, outwearData,outwearName)
    mergeAndSave(cocoDataFOutwear, savepath)

    cocoDataFDress =CocoAndSave(basepath, dressData,dressName)
    mergeAndSave(cocoDataFDress, savepath)
    cocoDataFSkirt =CocoAndSave(basepath, skirtData,skirtName)
    mergeAndSave(cocoDataFSkirt, savepath)
    cocoDataFTrouser =CocoAndSave(basepath, trouserData,trouserName)
    mergeAndSave(cocoDataFTrouser, savepath)


def mergeAndSave(cocoDataFBlouse, savepath):
    merged_coco_data = {
        "info": {"version": "1.0", "description": "Merged custom datasets"},
        "licenses": [],
        "images": [],
        "annotations": [],
        "categories": []
    }
    category_name = cocoDataFBlouse[0].get('categories')[0].get('name')
    for m_coco_data in cocoDataFBlouse:
        merged_coco_data["images"].extend(m_coco_data["images"])
        merged_coco_data["annotations"].extend(m_coco_data["annotations"])
    merged_coco_data["categories"].extend(cocoDataFBlouse[0].get('categories'))
    # with open("merged_coco_data.json", "w") as json_file:
    #     json.dump(merged_coco_data, json_file, indent=2)
    json_string = json.dumps(merged_coco_data, indent=2)
    sa = os.path.join(savepath, category_name + ".json")
    with open(sa, "w") as json_file:
        json_file.write(json_string)


def CocoAndSave(basepath, blouseData,blouseName):
    coco_datafirst = []
    category_name = ""

    hash_to_id_mapping = {hashlib.sha256(img_name.encode()).hexdigest(): i for i, img_name in enumerate(blouseName)}

    # 创建整数ID到哈希值的映射字典
    id_to_hash_mapping = {i: hashlib.sha256(img_name.encode()).hexdigest() for i, img_name in enumerate(blouseName)}
    anno_id = 0
    for blouseDataSingle in blouseData:
        coco_data = {
            "info": {"version": "1.0", "description": "My custom dataset"},
            "licenses": [],
            "images": [],
            "annotations": [],
            "categories": []
        }
        image_fileName = blouseDataSingle.get("image_fileName")
        category_name = blouseDataSingle.get("category_name")
        name = image_fileName.split(".")[0]
        keypoints_data = blouseDataSingle.get("keypoints_data")
        category_info = {"id":"1","name": category_name,
                         "keypoints": [keypoint['name'] for keypoint in keypoints_data]}
        coco_data["categories"].append(category_info)
        # 添加图像信息
        ##读一下图片获取一下size吧 把框打上去
        image_path = os.path.join(basepath, "Images", category_name, image_fileName)
        with Image.open(image_path) as img:
            width, height = img.size
        #图片的id需要转为整数
        image_info = {
            "id": hash_to_id_mapping.get(hashlib.sha256(name.encode()).hexdigest()),
            "file_name": image_fileName,
            "width": width,  # 请替换为实际图像宽度
            "height": height  # 请替换为实际图像高度
        }
        coco_data["images"].append(image_info)
       # for i, keypoint_data in enumerate(keypoints_data, start=1):
        annotation_info = {
            "id": anno_id, #设置自增主键
            "image_id": hash_to_id_mapping.get(hashlib.sha256(name.encode()).hexdigest()),
            "category_id": "1",
            "bbox": [0, 0, width, height],  # 请根据实际情况提供边界框信息
            "iscrowd": 0,  # 通常设置为0，表示非群体标注
            "area": width * height,  # 请根据实际情况提供标注区域面积信息
            "keypoints": [[keypoint_data["coordinates"][0], keypoint_data["coordinates"][1],
                           keypoint_data["visibility"]] for keypoint_data in keypoints_data]
        }
        anno_id += 1
        coco_data["annotations"].append(annotation_info)
        coco_datafirst.append(coco_data)
    return coco_datafirst



def getKeyPoint(listBlouse, pathVal, row, sortVal):
    keypoint_data = []
    for index in listBlouse:
        val = row[index + 2]
        keyname = reverse_mapping[index]  # 获取keyname 下一步开始组装
        x, y, visiable = 0, 0, 0
        ##防止应该有的点是-1 还是一个一个取吧
        if val == '-1':
            print(sortVal + 'a wrong data in' + keyname)
        else:
            x, y, visiable = val.split('_')
        singelKeypoint_data = {"name": keyname, "coordinates": (x, y), "visibility": visiable}
        keypoint_data.append(singelKeypoint_data)
    blouseDataSingle = {"image_fileName": pathVal, "category_name": sortVal, "keypoints_data": keypoint_data}
    return blouseDataSingle


if __name__ == "__main__":
    readCsvToJson()
