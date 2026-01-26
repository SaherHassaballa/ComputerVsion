from roboflow import Roboflow

model_path = "c:/Users/saher/Desktop/github/cv_projects/Identifying and locating flags from aerial photography/src/models/trained_models/flags_yolov8x_v1"
filename = "weights/best.pt"

rf = Roboflow(api_key="NWvWInlXkKL7xoWrS7XD")
project = rf.workspace("hassaballa").project("detect_isflage")
version = project.version(1)
version.deploy(
    model_type="yolov8x",
    model_path="src/models/trained_models/flags_yolov8x_v1",
    filename="weights/best.pt",
)

print("Model deployed successfully!")
