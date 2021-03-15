python -m pip cache remove opencv-python
python -m pip cache purge
pip install git+file:///netscratch/minouei/report/tf/TensorFlow-Advanced-Segmentation-Models
python slurm_deeplabv3.py
