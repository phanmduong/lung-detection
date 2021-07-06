from math import log
import pandas as pd
import numpy as np
from ast import literal_eval

NEG_PATH = "./data_preprocessed/augmented/negatives/"
POS_PATH = "./data_preprocessed/augmented/positives/"

def getData(path):
    all_imgs = {}
    classes_count = {};
    class_mapping = {0: "0", 1: "1"}
    meta = pd.read_csv(path, index_col=0);
    series = meta['seriesuid'].tolist()
    for index, row in meta.iterrows():

        class_name = row['class']

        if class_name not in classes_count:
            classes_count[class_name] = 1
        else:
            classes_count[class_name] += 1

        filename = row['seriesuid'] + "_" + row['sub_index'] + ".npy"
        basepath = POS_PATH if row['class'] == 1 else NEG_PATH
        file_path = basepath + filename;
        img = np.load(file=f'{file_path}')
        (rows, cols, depths) = img.shape[:3]
        all_imgs[filename] = {
            "filepath": filename,
            "bboxes": [],
            "width": cols,
            "height": rows,
            "deep" : depths
        }
        listRadii = literal_eval(row['radii']);
        indexCenter = 0
        for center in literal_eval(row['centers']):
        #     print(row['seriesuid']);
            radii = listRadii[indexCenter]
            all_imgs[filename]['bboxes'].append({
                "class": row['class'],
                "x1": round(center[0] - radii),
                "y1": round(center[1] - radii),
                "z1": round(center[2] - radii),
                "x2": round(center[0] + radii),
                "y2": round(center[1] + radii),
                "z2": round(center[2] + radii),
            })
            indexCenter = indexCenter + 1
    
    all_data = []
    for key in all_imgs:
        all_data.append(all_imgs[key])
    
    return all_data, classes_count, class_mapping
    

# all_data, classes_count, class_mapping = getData("./data_preprocessed/augmented_meta2.csv");
