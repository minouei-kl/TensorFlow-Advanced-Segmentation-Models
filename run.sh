pip3 install opencv-python==4.1.2.30
apt update && apt install -y libsm6 libxext6
pip install git+file:///netscratch/minouei/report/tf/TensorFlow-Advanced-Segmentation-Models
python slurm_deeplabv3.py
