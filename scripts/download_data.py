"""从 Roboflow 下载车牌识别数据集（在 AutoDL Linux 上运行）"""
from roboflow import Roboflow

API_KEY = "h5MQcuqAwwpVtLrozKa9"

rf = Roboflow(api_key=API_KEY)
project = rf.workspace("roboflow-universe-projects").project("license-plate-recognition-rxg4e")
dataset = project.version(11).download("yolov8", location="data/dataset")

print(f"数据集已下载到: data/dataset/")
