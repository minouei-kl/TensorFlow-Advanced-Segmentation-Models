import os
import cv2
import numpy as np
import tensorflow as tf
from PIL import Image



TOTAL_CLASSES = ['background', 'headerlogo', 'twocoltabel', 'recieveraddress', 'text', 'senderaddress', 'ortdatum',
 'companyinfo', 'fulltabletyp1', 'fulltabletyp2', 'copylogo', 'footerlogo', 'footertext', 'signatureimage', 'fulltabletyp3']

MODEL_CLASSES = TOTAL_CLASSES
N_CLASSES = 15
HEIGHT = 704
WIDTH = 704



def OverLayLabelOnImage(ImgIn,Label,W=0.6):
    #ImageIn is the image
    #Label is the label per pixel
    # W is the relative weight in which the labels will be marked on the image
    # Return image with labels marked over it
    Img=ImgIn.copy()
    TR = [0,1, 0, 0,   0, 1, 1, 0, 0,   0.5, 0.7, 0.3, 0.5, 1,    0.5, 0.3]
    TB = [0,0, 1, 0,   1, 0, 1, 0, 0.5, 0,   0.2, 0.2, 0.7, 0.5,  1,   0.3]
    TG = [0,0, 0, 0.5, 1, 1, 0, 1, 0.7, 0.4, 0.7, 0.2, 0,   0.25, 0.5, 0.3]
    R = Img[:, :, 0].copy()
    G = Img[:, :, 1].copy()
    B = Img[:, :, 2].copy()
    for i in range(Label.max()+1):
        if i<len(TR): #Load color from Table
           R[Label == i] = TR[i] * 255
           G[Label == i] = TG[i] * 255
           B[Label == i] = TB[i] * 255
        else: #Generate random label color
           R[Label == i] = np.mod(i*i+4*i+5,255)
           G[Label == i] = np.mod(i*10,255)
           B[Label == i] = np.mod(i*i*i+7*i*i+3*i+30,255)
    Img[:, :, 0] = Img[:, :, 0] * (1 - W) + R * W
    Img[:, :, 1] = Img[:, :, 1] * (1 - W) + G * W
    Img[:, :, 2] = Img[:, :, 2] * (1 - W) + B * W
    return Img

def save_seg_result(image, pred_mask, gt_mask=None, image_id=1):
    # save predict mask as PNG image
    mask_dir = os.path.join('result','predict_mask')
    os.makedirs(mask_dir, exist_ok=True)
    result_dir = os.path.join('result','segmentation')
    os.makedirs(result_dir, exist_ok=True)

    im = Image.fromarray(pred_mask)
    im.save(os.path.join(mask_dir, str(image_id)+'.png'))

    im = Image.fromarray(OverLayLabelOnImage(image,pred_mask).astype('uint8'))
    im.save(os.path.join(result_dir, str(image_id)+'.png'))
    
    if gt_mask is not None:
        im = Image.fromarray(OverLayLabelOnImage(image,gt_mask).astype('uint8'))
        im.save(os.path.join(result_dir, str(image_id)+'_GT.png'))

class BilinearUpsampling(tf.keras.layers.Layer):

    def __init__(self, upsampling=(2, 2), data_format=None, **kwargs):
        super(BilinearUpsampling, self).__init__(**kwargs)
        self.data_format = tf.python.keras.utils.conv_utils.normalize_data_format(data_format)
        self.upsampling = tf.python.keras.utils.conv_utils.normalize_tuple(upsampling, 2, 'size')
        self.input_spec = tf.keras.layers.InputSpec(ndim=4)

    def compute_output_shape(self, input_shape):
        height = self.upsampling[0] * \
                 input_shape[1] if input_shape[1] is not None else None
        width = self.upsampling[1] * \
                input_shape[2] if input_shape[2] is not None else None
        return (input_shape[0], height, width, input_shape[3])

    def call(self, inputs):
        # .tf
        return tf.image.resize(inputs, (int(inputs.shape[1] * self.upsampling[0]),
                                                 int(inputs.shape[2] * self.upsampling[1])), method=tf.image.ResizeMethod.BILINEAR)

    def get_config(self):

        config = {'upsampling': self.upsampling, 'data_format': self.data_format}
        base_config = super(BilinearUpsampling, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))

model = tf.keras.models.load_model('modelep2.h5',custom_objects={'BilinearUpsampling':BilinearUpsampling})

im_name='axa1-50653_00'

image = cv2.imread(im_name+'.jpg')
image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

image = tf.image.resize(image,(HEIGHT,WIDTH))
image = image.numpy().astype('uint8')

pr_mask = model.predict(np.expand_dims(image, axis=0))
pr_mask = np.argmax(pr_mask, axis=-1)
pr_mask = pr_mask.squeeze().astype('uint8')

gt_mask = None
if os.path.isfile(im_name+'.png'):
    gt_mask = cv2.imread(im_name+'.png', 0)
    gt_mask = tf.image.resize(np.expand_dims(gt_mask, axis=-1),(HEIGHT,WIDTH))
    gt_mask = gt_mask.numpy().squeeze().astype('uint8')

save_seg_result(image,pr_mask,gt_mask,im_name)

