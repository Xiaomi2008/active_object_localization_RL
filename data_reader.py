import xml.dom.minidom
from xml.etree import ElementTree as ET
import skimage.io as io
from skimage import data_dir
import random
import os
###========annotations==========###
def Get_file_inference(location):
    path = location
    filelist = os.listdir(path)
    return filelist

def Get_Annotation(location):
    per=ET.parse(location)
    p=per.findall('./object')
    Annotations_name = []
    for oneper in p:
        sublistAnnotation = []
        for child in oneper.getchildren():
            if child.tag == 'name':
                sublistAnnotation.append(child.text)
        Annotations_name.append(sublistAnnotation)
    p=per.findall('./object/bndbox')
    Annotations = []
    for oneper in p:
        sublistAnnotation = []
        for child in oneper.getchildren():
            child.text = int(child.text)
            sublistAnnotation.append(child.text)
        Annotations.append(sublistAnnotation)
    return Annotations,Annotations_name
###=====Image======###
def Get_Image(location):
    coll = io.ImageCollection(location)
    return coll

def get_dataset_size(Annotations_loc,Image_loc):
    labellist = Get_file_inference(Annotations_loc)
    imagelist = Get_file_inference(Image_loc)
    if len(labellist) == len(imagelist):
        return len(labellist)
    else:
        return 'Image size does not match with the Annotations '

def main(Annotations_loc,Image_loc,i):
    filelist = Get_file_inference(Annotations_loc)
    Anno_Loc = Annotations_loc + filelist[i]
    Annotations,Annotations_name=Get_Annotation(Anno_Loc)
    image_loc = Image_loc+'*.jpg'
    coll=Get_Image(image_loc)
    return Annotations,Annotations_name, coll[i]

if __name__ == "__main__":
    Annotations_loc = 'Annotations/'
    Image_loc = 'JPEGImages/'
    max_num = get_dataset_size(Annotations_loc,Image_loc)
    if isinstance(max_num, int):
        index = random.randint(0, max_num)
        label, label_name,iamge = main(Annotations_loc,Image_loc,index)
    else:
        print(max_num)
    print(label, label_name,iamge.shape)








