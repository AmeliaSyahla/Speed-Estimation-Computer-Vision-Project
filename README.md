# Speed-Estimation-Computer-Vision-Project
Ensure you have download python ver > 3.8 !!!
## Step for run the code:

1. Clone repository
  ```bash
  git clone https://github.com/Amelia-Syahla/Speed-Estimation-Computer-Vision-Project.git
  cd Speed-Estimation-Computer-Vision-Project
  ```
2. Activate your virtual environment
   Windows:
   ```bash
   python -m venv venv
   venv\Scripts\activate
   ```
   Linux:
   ```bash
   python3 -m venv venv
   source venv/bin/activate
   ```
3. Install dependencies
   ```bash
   pip install --upgrade pip
   pip install -r requirements.txt
   ```
4. Run code in your teminal
   ```bash
   python filename.py
   ```
   Dont forget to save your code before run, or you will get some error.

## Notes
1. You must download video dataset before you run all 
2. File ```01_extract_frames.py``` contains step to make Dataset/ format video to frame. It will use for speed estimation. When you run this, the code will make new folder 'frames'
3. File ``02_annotate_calib.py`` contains step to make you annotating through point and line to calibration and homography. This code will make new folder ``
4. File ``03_calib_homography.py`` contains step to connect a single point from the previous annotation. This code will make new folder ``
5. You can use SSD Model or Faster-RCNN to detection object and tracking, when you want to use Faster-RCNN, just run `04_detectcnn.py`. This code will automatically downloaded model .pth and make new folder `Output_Json_Data` and `Output_Videos`
6. To look the output, you can access it on your local folder
