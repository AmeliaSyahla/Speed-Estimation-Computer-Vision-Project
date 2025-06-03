from roboflow import Roboflow
rf = Roboflow(api_key="PqYSsHkElB0oqd2AMtlC")
project = rf.workspace("cv-kboj1").project("cv-speed-estimation")
version = project.version(1)
dataset = version.download("voc")
                