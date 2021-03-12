import tensorflow_advanced_segmentation_models as tasm
import os
import cv2
import numpy as np
from time import time
import tensorflow as tf
import albumentations as A
import matplotlib.pyplot as plt
import tensorflow.keras.backend as K
from labelme.utils import lblsave as label_save
from common.utils import  visualize_segmentation
from PIL import Image

DATA_DIR = "/home/minouei/Downloads/datasets/contract/version2"

x_train_dir = os.path.join(DATA_DIR, 'images/train')
y_train_dir = os.path.join(DATA_DIR, 'annotations/train')

x_valid_dir = os.path.join(DATA_DIR, 'images/val')
y_valid_dir = os.path.join(DATA_DIR, 'annotations/val')

x_test_dir = os.path.join(DATA_DIR, 'images/val')
y_test_dir = os.path.join(DATA_DIR, 'annotations/val')

"""### Helper Functions"""

# helper function for data visualization
def visualize(**images):
    """PLot images in one row."""
    n = len(images)
    plt.figure(figsize=(16, 5))
    ns=''
    for i, (name, image) in enumerate(images.items()):
        plt.subplot(1, n, i + 1)
        plt.xticks([])
        plt.yticks([])
        plt.title(' '.join(name.split('_')).title())
        plt.imshow(image)
        ns=f'{name}_{i}'
    # plt.show()
    plt.savefig(ns+'.png', bbox_inches='tight')
    
# helper function for data visualization    
def denormalize(x):
    """Scale image to range 0..1 for correct plot"""
    x_max = np.percentile(x, 98)
    x_min = np.percentile(x, 2)    
    x = (x - x_min) / (x_max - x_min)
    x = x.clip(0, 1)
    return x
    
def round_clip_0_1(x, **kwargs):
    return x.round().clip(0, 1)

"""## Data Augmentation Functions"""

# define heavy augmentations
def get_training_augmentation(height, width):
    train_transform = [

        A.PadIfNeeded(min_height=height, min_width=width, always_apply=True, border_mode=0),
        A.Resize(height, width, always_apply=True)
    ]
    return A.Compose(train_transform)


def get_validation_augmentation(height, width):
    """Add paddings to make image shape divisible by 32"""
    test_transform = [
        A.PadIfNeeded(height, width),
        A.Resize(height, width, always_apply=True)
    ]
    return A.Compose(test_transform)

def data_get_preprocessing(preprocessing_fn):
    """Construct preprocessing transform
    
    Args:
        preprocessing_fn (callbale): data normalization function 
            (can be specific for each pretrained neural network)
    Return:
        transform: albumentations.Compose
    
    """
    
    _transform = [
        A.Lambda(image=preprocessing_fn),
    ]
    return A.Compose(_transform)

"""## Define some global variables"""

TOTAL_CLASSES = ['background', 'headerlogo', 'twocoltabel', 'recieveraddress', 'text', 'senderaddress', 'ortdatum',
 'companyinfo', 'fulltabletyp1', 'fulltabletyp2', 'copylogo', 'footerlogo', 'footertext', 'signatureimage', 'fulltabletyp3', 'unlabelled']


MODEL_CLASSES = TOTAL_CLASSES
ALL_CLASSES = False
if MODEL_CLASSES == TOTAL_CLASSES:
    MODEL_CLASSES = MODEL_CLASSES[:-1]
    ALL_CLASSES = True

BATCH_SIZE = 1
N_CLASSES = 16
HEIGHT = 704
WIDTH = 704
BACKBONE_NAME = "efficientnetb3"
WEIGHTS = "imagenet"
WWO_AUG = False # train data with and without augmentation

"""### Functions to calculate appropriate class weights"""

################################################################################
# Class Weights
################################################################################
def get_dataset_counts(d):
    pixel_count = np.array([i for i in d.values()])

    sum_pixel_count = 0
    for i in pixel_count:
        sum_pixel_count += i

    return pixel_count, sum_pixel_count

def get_dataset_statistics(pixel_count, sum_pixel_count):
    
    pixel_frequency = np.round(pixel_count / sum_pixel_count, 4)

    mean_pixel_frequency = np.round(np.mean(pixel_frequency), 4)

    return pixel_frequency, mean_pixel_frequency

def get_balancing_class_weights(classes, d):
    pixel_count, sum_pixel_count = get_dataset_counts(d)

    background_pixel_count = 0
    mod_pixel_count = []
    for c in TOTAL_CLASSES:
        if c not in classes:
            background_pixel_count += d[c]
        else:
            mod_pixel_count.append(d[c])
    mod_pixel_count.append(background_pixel_count)
    
    pixel_frequency, mean_pixel_frequency = get_dataset_statistics(mod_pixel_count, sum_pixel_count)

    class_weights = np.round(mean_pixel_frequency / pixel_frequency, 2)
    return class_weights    

# class_weights = get_balancing_class_weights(MODEL_CLASSES, CLASSES_PIXEL_COUNT_DICT)
# print(class_weights)
def save_seg_result(image, pred_mask, gt_mask, image_id, class_names):
    # save predict mask as PNG image
    mask_dir = os.path.join('result','predict_mask')
    os.makedirs(mask_dir, exist_ok=True)
    label_save(os.path.join(mask_dir, str(image_id)+'.png'), pred_mask)

    # visualize segmentation result
    title_str = 'Predict Segmentation\nmIOU: ' 
    gt_title_str = 'GT Segmentation'
    image_array = visualize_segmentation(image, pred_mask, gt_mask, class_names=class_names, title=title_str, gt_title=gt_title_str, ignore_count_threshold=1)

    # save result as JPG
    result_dir = os.path.join('result','segmentation')
    os.makedirs(result_dir, exist_ok=True)
    result_file = os.path.join(result_dir, str(image_id)+'.jpg')
    Image.fromarray(image_array).save(result_file)

"""## Data Generation Functions"""

################################################################################
# Data Generator
################################################################################
def create_image_label_path_generator(images_dir, masks_dir, shuffle=False, seed=None):
    ids = sorted(os.listdir(images_dir))
    mask_ids = sorted(os.listdir(masks_dir))

    if shuffle == True:

        if seed is not None:
            tf.random.set_seed(seed)

        indices = tf.range(start=0, limit=tf.shape(ids)[0], dtype=tf.int32)
        shuffled_indices = tf.random.shuffle(indices)

        ids = tf.gather(ids, shuffled_indices).numpy().astype(str)
        mask_ids = tf.gather(mask_ids, shuffled_indices).numpy().astype(str)

    images_fps = [os.path.join(images_dir, image_id) for image_id in ids]
    masks_fps = [os.path.join(masks_dir, image_id) for image_id in mask_ids]

    while True:
        for i in range(len(images_fps)):
            yield [images_fps[i], masks_fps[i]]


def process_image_label(images_paths, masks_paths, classes, augmentation=None, preprocessing=None):
    class_values = [TOTAL_CLASSES.index(cls.lower()) for cls in classes]
    
    # read data
    image = cv2.imread(images_paths)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    mask = cv2.imread(masks_paths, 0)

    # extract certain classes from mask (e.g. cars)
    masks = [(mask == v) for v in class_values]
    mask = np.stack(masks, axis=-1).astype('float')
    
    # add background if mask is not binary
    if mask.shape[-1] != 1:
        background = 1 - mask.sum(axis=-1, keepdims=True)
        mask = np.concatenate((mask, background), axis=-1)
    
    # apply augmentations
    if augmentation:
        sample = augmentation(image=image, mask=mask)
        image, mask = sample['image'], sample['mask']
    
    # apply preprocessing
    if preprocessing:
        sample = preprocessing(image=image, mask=mask)
        image, mask = sample['image'], sample['mask']

    # mask = np.squeeze(np.argmax(mask, axis=-1))
    # mask = np.argmax(mask, axis=-1)
    # mask = mask[..., np.newaxis]
        
    return image, mask

def DataGenerator(train_dir, label_dir, batch_size, height, width, classes, augmentation, wwo_aug=False, shuffle=False, seed=None):
    image_label_path_generator = create_image_label_path_generator(
        train_dir, label_dir, shuffle=shuffle, seed=seed
    )
    if wwo_aug:
        while True:
            images = np.zeros(shape=[batch_size, height, width, 3])
            labels = np.zeros(shape=[batch_size, height, width, len(classes) + 1], dtype=np.float32)
            for i in range(0, batch_size, 2):
                image_path, label_path = next(image_label_path_generator)
                image_aug, label_aug = process_image_label(image_path, label_path, classes=classes, augmentation=augmentation)
                image_wo_aug, label_wo_aug = process_image_label(image_path, label_path, classes=classes, augmentation=get_validation_augmentation(height=HEIGHT, width=WIDTH))
                images[i], labels[i] = image_aug, label_aug
                images[i + 1], labels[i + 1] = image_wo_aug, label_wo_aug

            yield tf.convert_to_tensor(images), tf.convert_to_tensor(labels, tf.float32)
    else:
        while True:
            images = np.zeros(shape=[batch_size, height, width, 3])
            labels = np.zeros(shape=[batch_size, height, width, len(classes) + 1], dtype=np.float32)
            for i in range(batch_size):
                image_path, label_path = next(image_label_path_generator)
                image, label = process_image_label(image_path, label_path, classes=classes, augmentation=augmentation)
                images[i], labels[i] = image, label

            yield tf.convert_to_tensor(images), tf.convert_to_tensor(labels, tf.float32)

"""## Create the Model"""

base_model, layers, layer_names = tasm.create_base_model(name=BACKBONE_NAME, weights=WEIGHTS, height=HEIGHT, width=WIDTH, include_top=False, pooling=None)

BACKBONE_TRAINABLE = True
model = tasm.DANet(n_classes=N_CLASSES, base_model=base_model, output_layers=layers, backbone_trainable=BACKBONE_TRAINABLE)

model.load_weights("DANet.ckpt")


def display(display_list):
    plt.figure(figsize=(15, 15))

    title = ['Input Image', 'True Mask', 'Predicted Mask']

    for i in range(len(display_list)):
        plt.subplot(1, len(display_list), i+1)
        plt.title(title[i])
        plt.imshow(tf.keras.preprocessing.image.array_to_img(display_list[i]))
        plt.axis('off')
    plt.show()

def create_mask(pred_mask):
    
    pred_mask = tf.argmax(pred_mask, axis=-1)
    pred_mask = pred_mask[..., tf.newaxis]
    return pred_mask[0]

def show_predictions(dataset=None, num=1):
    
    output_model = model(sample_image[tf.newaxis, ...])
    # print(output_model.numpy())
    
    output_mask = create_mask(output_model)
    # print(sample_mask.shape)

    scce = tf.keras.losses.CategoricalCrossentropy()
    print("SparseCategoricalCrossentroy: " + str(scce(sample_mask, output_model[0]).numpy()))
    print("Iou-Score: " + str(tasm.losses.iou_score(sample_mask, output_model[0]).numpy()))
    print("categorical Focal Dice Loss: " + str(categorical_focal_dice_loss(sample_mask, output_model[0]).numpy()))
    
    display([sample_image, sample_mask, K.one_hot(K.squeeze(output_mask, axis=-1), 3)])
    
# show_predictions()
val_shuffle = True
seed = 29598

ValidationSet = DataGenerator(
    x_valid_dir,
    y_valid_dir,
    1,
    HEIGHT,
    WIDTH,
    classes=MODEL_CLASSES,
    augmentation=get_validation_augmentation(height=HEIGHT, width=WIDTH),
    shuffle=val_shuffle,
    seed=seed
)

"""# Evaluation on Test Data"""

# scores = model.evaluate(TestSet, steps=101)

# print("Loss: {:.5}".format(scores[0]))
# for metric, value in zip(metrics, scores[1:]):
#     if metric != "accuracy":
#         metric = metric.__name__
#     print("mean {}: {:.5}".format(metric, value))

# """## Visual Examples on Test Data"""

n = 5
ids = np.random.choice(np.arange(101), size=n,replace=False)
# print(ids)

counter = 0
second_counter = 0
for i in ValidationSet:
    if counter in ids:
        image, gt_mask = i
        # image = np.expand_dims(image, axis=0)
        pr_mask = model.predict(image)
        pr_mask = np.argmax(pr_mask, axis=-1)
        gt_mask = np.argmax(gt_mask, axis=-1)
        print(pr_mask.shape)

        print(counter)
        # cv2.imwrite(f'{counter}.jpg',denormalize(image.numpy().squeeze()))
        # cv2.imwrite(f'{counter}.png',pr_mask.squeeze())
        image=denormalize(image.numpy().squeeze())
        gt_mask=gt_mask.squeeze()

        print(gt_mask.shape)
        pr_mask=pr_mask.squeeze()
        print(pr_mask.shape)

        save_seg_result(image,pr_mask,gt_mask,counter,TOTAL_CLASSES)
        # visualize(
        #     image=denormalize(image.numpy().squeeze()),
        #     # gt_mask=gt_mask.numpy().squeeze(),
        #     pr_mask=pr_mask.squeeze(),
        # )
        second_counter += 1
    counter += 1
    if second_counter == n:
        break

