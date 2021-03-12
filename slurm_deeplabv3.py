import tensorflow_advanced_segmentation_models as tasm
import os
import cv2
import numpy as np
from time import time
import tensorflow as tf
import albumentations as A
import matplotlib.pyplot as plt
import tensorflow.keras.backend as K
from functools import partial

DATA_DIR = "/netscratch/minouei/versicherung/version2"

x_train_dir = os.path.join(DATA_DIR, 'images/train')
y_train_dir = os.path.join(DATA_DIR, 'annotations/train')

x_valid_dir = os.path.join(DATA_DIR, 'images/val')
y_valid_dir = os.path.join(DATA_DIR, 'annotations/val')

x_test_dir = os.path.join(DATA_DIR, 'images/val')
y_test_dir = os.path.join(DATA_DIR, 'annotations/val')


"""## Data Augmentation Functions"""

def get_validation_augmentation(height, width):
    """Add paddings to make image shape divisible by 32"""
    test_transform = [
        A.PadIfNeeded(height, width),
        A.Resize(height, width, always_apply=True)
    ]
    return A.Compose(test_transform)


"""## Define some global variables"""

TOTAL_CLASSES = ['background', 'headerlogo', 'twocoltabel', 'recieveraddress', 'text', 'senderaddress', 'ortdatum',
 'companyinfo', 'fulltabletyp1', 'fulltabletyp2', 'copylogo', 'footerlogo', 'footertext', 'signatureimage', 'fulltabletyp3', 'unlabelled']


MODEL_CLASSES = TOTAL_CLASSES
ALL_CLASSES = False
if MODEL_CLASSES == TOTAL_CLASSES:
    MODEL_CLASSES = MODEL_CLASSES[:-1]
    ALL_CLASSES = True

BATCH_SIZE = 4
N_CLASSES = 16
HEIGHT = 704
WIDTH = 704
BACKBONE_NAME = "efficientnetb3"
WEIGHTS = "imagenet"
WWO_AUG = False # train data with and without augmentation


"""## Data Generation Functions"""

################################################################################
# Data Generator
################################################################################
def get_filtered(dir):
    included_extensions = ['jpg','jpeg', 'png',]
    file_names = [fn for fn in os.listdir(dir)
                if any(fn.endswith(ext) for ext in included_extensions)]
    return sorted(file_names)

def create_image_label_path_generator(images_dir, masks_dir):
    ids = get_filtered(images_dir)
    mask_ids = get_filtered(masks_dir)

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

def DataGenerator(train_dir, label_dir, height, width, classes, augmentation):
    image_label_path_generator = create_image_label_path_generator(
        train_dir, label_dir)
    while True:
        images = np.zeros(shape=[height, width, 3])
        labels = np.zeros(shape=[height, width, len(classes) + 1], dtype=np.float32)
        image_path, label_path = next(image_label_path_generator)
        image, label = process_image_label(image_path, label_path, classes=classes, augmentation=augmentation)
        images , labels  = image, label
        yield tf.convert_to_tensor(images), tf.convert_to_tensor(labels, tf.float32)


TrainSetwoAug = partial(DataGenerator,
    x_train_dir,
    y_train_dir,
    HEIGHT,
    WIDTH,
    classes=MODEL_CLASSES,
    augmentation=get_validation_augmentation(height=HEIGHT, width=WIDTH),
)

ValidationSet =partial(DataGenerator,
    x_valid_dir,
    y_valid_dir,
    HEIGHT,
    WIDTH,
    classes=MODEL_CLASSES,
    augmentation=get_validation_augmentation(height=HEIGHT, width=WIDTH),
)

TrainSet = tf.data.Dataset.from_generator(
    TrainSetwoAug,
    (tf.float32, tf.float32),
    (tf.TensorShape([None, None, 3]), tf.TensorShape([None, None,N_CLASSES]))
).batch(BATCH_SIZE, drop_remainder=True)

ValSet = tf.data.Dataset.from_generator(
    ValidationSet,
    (tf.float32, tf.float32),
    (tf.TensorShape([None, None, 3]), tf.TensorShape([None, None,N_CLASSES]))
).batch(BATCH_SIZE, drop_remainder=True)


# mirrored_strategy = tf.distribute.experimental.MultiWorkerMirroredStrategy(
#     cluster_resolver=tf.distribute.cluster_resolver.SlurmClusterResolver(port_base=15000),
#     communication=tf.distribute.experimental.CollectiveCommunication.NCCL,
# )

slurm_resolver = tf.distribute.cluster_resolver.SlurmClusterResolver(port_base=15000)
# communication_options = tf.distribute.experimental.CommunicationOptions(
#             implementation=tf.distribute.experimental.CommunicationImplementation.AUTO)
mirrored_strategy = tf.distribute.MultiWorkerMirroredStrategy(cluster_resolver=slurm_resolver )
                                                        # ,communication_options=communication_options)



with mirrored_strategy.scope():
    print('----------------------mirrored_strategy.num_replicas_in_sync')
    print(mirrored_strategy.num_replicas_in_sync)

    train_dist_dataset = mirrored_strategy.experimental_distribute_dataset(TrainSet)
    val_dist_dataset = mirrored_strategy.experimental_distribute_dataset(ValSet)

    base_model, layers, layer_names = tasm.create_base_model(name=BACKBONE_NAME, weights=WEIGHTS, height=HEIGHT, width=WIDTH, include_top=False, pooling=None)
    model = tasm.DeepLabV3plus(n_classes=N_CLASSES, base_model=base_model, output_layers=layers, backbone_trainable=True)

    for layer in model.layers:
        layer.trainable = True
        print(layer.name + ": " + str(layer.trainable))

    opt = tf.keras.optimizers.SGD(learning_rate=0.1, momentum=0.9)
    metrics = [tasm.metrics.IOUScore(threshold=0.5)]
    categorical_focal_dice_loss = tasm.losses.CategoricalFocalLoss(alpha=0.25, gamma=2.0) + tasm.losses.DiceLoss()

    model.compile(
        optimizer=opt,
        loss=categorical_focal_dice_loss,
        metrics=metrics,
    )
    model.run_eagerly = False

callbacks = [
            #  tf.keras.callbacks.ModelCheckpoint("DeepLabV3plus.ckpt", verbose=1, save_weights_only=True, save_best_only=True),
             tf.keras.callbacks.experimental.BackupAndRestore(backup_dir='./backup'),
             tf.keras.callbacks.ReduceLROnPlateau(monitor="val_iou_score", factor=0.2, patience=6, verbose=1, mode="max"),
             tf.keras.callbacks.EarlyStopping(monitor="val_iou_score", patience=16, mode="max", verbose=1, restore_best_weights=True)
]

steps_per_epoch = np.floor(len(os.listdir(x_train_dir)) / BATCH_SIZE)

history = model.fit(
    train_dist_dataset,
    steps_per_epoch=steps_per_epoch,
    epochs=2,
    callbacks=callbacks,
    validation_data=val_dist_dataset,
    validation_steps=len(os.listdir(x_valid_dir)),
    )

model.save("model1.h5")