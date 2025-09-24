from roboflow import Roboflow

rf = Roboflow(api_key="DLhVJCl8QLr7KygV36XO")
project = rf.workspace("pothole-detection-y1mqk").project("pothole-detection-i09mh")
version = project.version(2)
dataset = version.download("yolov9")
print(dataset)
