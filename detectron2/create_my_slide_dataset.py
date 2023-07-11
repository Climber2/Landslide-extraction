from detectron2.structures import BoxMode
from detectron2.data import MetadataCatalog, DatasetCatalog
from detectron2.utils.visualizer import Visualizer
import os
import json
import cv2
import numpy as np
import random

def get_slide_dicts(img_dir,annotation_dir):
    dataset_dicts = []
    json_file_names= os.listdir(annotation_dir) #得到anns文件夹下的所有json文件名称
    for idx in range(len(json_file_names)):
        record = {}
        json_file = os.path.join(annotation_dir, json_file_names[idx])
        with open(json_file) as f:
            info = json.load(f)
        img_file = os.path.join(img_dir, info['filename'])
        height, width = cv2.imread(img_file).shape[:2]
        record["file_name"] = img_file
        record["image_id"] = idx
        record["height"] = height
        record["width"] = width
        
        annos = info['regions']
        objs = []
        # 对于单张图片的每个标注（annotation）
        for _, anno in annos.items():
            anno = anno["shape_attributes"]
            px = anno["all_points_x"]
            py = anno["all_points_y"]
            poly = [(x+0.5, y+0.5) for x, y in zip(px, py)]
            poly = [p for x in poly for p in x]
            obj = {
                "bbox":[np.min(px), np.min(py), np.max(px), np.max(py)], # 物体轮廓同时可以转换为一个框
                "bbox_mode": BoxMode.XYXY_ABS,
                "segmentation": [poly],
                "category_id": 0,
            }
            objs.append(obj)
        record["annotations"] = objs
        dataset_dicts.append(record)
    return dataset_dicts

# for d in ['train']:
#     DatasetCatalog.register("slide_"+d, lambda d=d: get_slide_dicts("../../samples/image","../../samples/annotation"))
#     MetadataCatalog.get("slide_"+d).set(things_classes=["slide"])

slide_metadata = MetadataCatalog.get("slide_train")

dataset_dicts = get_slide_dicts("../../samples/image","../../samples/annotation")
for d in random.sample(dataset_dicts, 1):
    img = cv2.imread(d["file_name"])
    visualizer = Visualizer(img[:, :, ::-1], metadata=slide_metadata, scale=0.5)
    out = visualizer.draw_dSataset_dict(d)
    cv2.imshow('slide',out.get_image()[:, :, ::-1])
    cv2.waitKey(0)