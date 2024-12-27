from ultralytics import YOLO

model = YOLO("/root/yourcfg.yaml")  # build a YOLOv8n model from scratch

model.info()  # display model information

model.train(data="/root/ultralytics/cfg/visdrone.yaml",
            epochs=100,
            name="yolo_mamba",
           imgsz=640,
           batch=16,
            device=0,
           project="/root/tf-logs/sota/visdrone")  # train the model
