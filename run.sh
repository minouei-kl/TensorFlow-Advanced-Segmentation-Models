pip install git+file:///netscratch/minouei/report/tf/TensorFlow-Advanced-Segmentation-Models
export LD_LIBRARY_PATH=/usr/local/lib/python3.8/dist-packages/opencv_python_headless.libs:$LD_LIBRARY_PATH
python slurm_deeplabv3.py
