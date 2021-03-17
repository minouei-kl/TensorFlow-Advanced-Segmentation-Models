import tensorflow_advanced_segmentation_models as tasm
import os
import cv2
import numpy as np
from time import time
import tensorflow as tf
# import albumentations as A
import matplotlib.pyplot as plt
import tensorflow.keras.backend as K
from functools import partial
from tensorflow.keras import mixed_precision
import math

DATA_DIR = "/netscratch/minouei/versicherung/version2"

# x_train_dir = os.path.join(DATA_DIR, 'images/train')
# y_train_dir = os.path.join(DATA_DIR, 'annotations/train')
x_train_dir = os.path.join(DATA_DIR, 'images/val')
y_train_dir = os.path.join(DATA_DIR, 'annotations/val')

x_valid_dir = os.path.join(DATA_DIR, 'images/val')
y_valid_dir = os.path.join(DATA_DIR, 'annotations/val')

x_test_dir = os.path.join(DATA_DIR, 'images/val')
y_test_dir = os.path.join(DATA_DIR, 'annotations/val')

TOTAL_CLASSES = ['background', 'headerlogo', 'twocoltabel', 'recieveraddress', 'text', 'senderaddress', 'ortdatum',
                 'companyinfo', 'fulltabletyp1', 'fulltabletyp2', 'copylogo', 'footerlogo', 'footertext',
                 'signatureimage', 'fulltabletyp3', 'unlabelled']

MODEL_CLASSES = TOTAL_CLASSES
ALL_CLASSES = False
if MODEL_CLASSES == TOTAL_CLASSES:
    MODEL_CLASSES = MODEL_CLASSES[:-1]
    ALL_CLASSES = True

BATCH_SIZE = 16
N_CLASSES = 16
HEIGHT = 640
WIDTH = 640

"""## Data Generation Functions"""


################################################################################
# Data Generator
################################################################################
def get_filtered(dir):
    included_extensions = ['jpg', 'jpeg', 'png', ]
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


def process_image_label(images_paths, masks_paths, classes):
    class_values = [TOTAL_CLASSES.index(cls.lower()) for cls in classes]

    # read data
    image = cv2.imread(images_paths)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    mask = cv2.imread(masks_paths, 0)

    image = cv2.resize(image, (HEIGHT, WIDTH), interpolation=cv2.INTER_AREA)
    mask = cv2.resize(mask, (HEIGHT, WIDTH), interpolation=cv2.INTER_AREA)

    # extract certain classes from mask (e.g. cars)
    masks = [(mask == v) for v in class_values]
    mask = np.stack(masks, axis=-1).astype('float')

    # image = tf.image.resize_with_pad(image,HEIGHT,WIDTH).numpy()
    # mask = tf.image.resize_with_pad(mask,HEIGHT,WIDTH).numpy()

    # add background if mask is not binary
    if mask.shape[-1] != 1:
        background = 1 - mask.sum(axis=-1, keepdims=True)
        mask = np.concatenate((mask, background), axis=-1)

    return image, mask


def DataGenerator(train_dir, label_dir, height, width, classes):
    image_label_path_generator = create_image_label_path_generator(
        train_dir, label_dir)
    while True:
        images = np.zeros(shape=[height, width, 3])
        labels = np.zeros(shape=[height, width, len(classes) + 1], dtype=np.float32)
        image_path, label_path = next(image_label_path_generator)
        image, label = process_image_label(image_path, label_path, classes=classes)
        images, labels = image, label
        yield tf.convert_to_tensor(images), tf.convert_to_tensor(labels, tf.float32)


def _is_chief(task_type, task_id):
    """Determines if the replica is the Chief."""
    return task_type is None or task_type == 'chief' or (
            task_type == 'worker' and task_id == 0)


def _get_saved_model_dir(base_path, task_type, task_id):
    """Returns a location for the SavedModel."""

    saved_model_path = base_path
    if not _is_chief(task_type, task_id):
        temp_dir = os.path.join('/tmp', task_type, str(task_id))
        tf.io.gfile.makedirs(temp_dir)
        saved_model_path = temp_dir

    return saved_model_path


TrainSetwoAug = partial(DataGenerator,
                        x_train_dir,
                        y_train_dir,
                        HEIGHT,
                        WIDTH,
                        classes=MODEL_CLASSES,
                        )

# ValidationSet =partial(DataGenerator,
#     x_valid_dir,
#     y_valid_dir,
#     HEIGHT,
#     WIDTH,
#     classes=MODEL_CLASSES,
# )


slurm_resolver = tf.distribute.cluster_resolver.SlurmClusterResolver(port_base=15000)
mirrored_strategy = tf.distribute.MultiWorkerMirroredStrategy(cluster_resolver=slurm_resolver)

print('----------------------mirrored_strategy.num_replicas_in_sync')
print(mirrored_strategy.num_replicas_in_sync)

TrainSet = tf.data.Dataset.from_generator(
    TrainSetwoAug,
    (tf.float32, tf.float32),
    (tf.TensorShape([None, None, 3]), tf.TensorShape([None, None, N_CLASSES]))
).batch(BATCH_SIZE, drop_remainder=True).prefetch(tf.data.AUTOTUNE)

# ValSet = tf.data.Dataset.from_generator(
#     ValidationSet,
#     (tf.float32, tf.float32),
#     (tf.TensorShape([None, None, 3]), tf.TensorShape([None, None,N_CLASSES]))
# ).batch(BATCH_SIZE, drop_remainder=True).prefetch(tf.data.AUTOTUNE)
options = tf.data.Options()
options.experimental_distribute.auto_shard_policy = tf.data.experimental.AutoShardPolicy.DATA
TrainSet = TrainSet.with_options(options)
# ValSet = ValSet.with_options(options)

train_dist_dataset = mirrored_strategy.experimental_distribute_dataset(TrainSet)
# val_dist_dataset = mirrored_strategy.experimental_distribute_dataset(ValSet)

with mirrored_strategy.scope():
    # mixed_precision.set_global_policy('mixed_float16')

    # base_model, layers, layer_names = tasm.create_base_model(name=BACKBONE_NAME, weights=WEIGHTS, height=HEIGHT, width=WIDTH, include_top=False, pooling=None)
    model = tasm.DeeplabV3_plus(N_CLASSES, HEIGHT, WIDTH)

    for layer in model.layers:
        layer.trainable = True
        # print(layer.name + ": " + str(layer.trainable))

    opt = tf.keras.optimizers.SGD(learning_rate=0.1, momentum=0.9)
    metrics = [tasm.metrics.IOUScore(threshold=0.5)]
    categorical_focal_dice_loss = tasm.losses.CategoricalFocalLoss(alpha=0.25, gamma=2.0) + tasm.losses.DiceLoss()

    model.compile(
        optimizer=opt,
        loss=categorical_focal_dice_loss,
        metrics=metrics,
    )
    # model.run_eagerly = False


# learning rate schedule
def step_decay(epoch):
    initial_lrate = 0.1
    drop = 0.5
    epochs_drop = 2.0
    lrate = initial_lrate * math.pow(drop, math.floor((1 + epoch) / epochs_drop))
    return lrate


callbacks = [
    tf.keras.callbacks.TensorBoard(log_dir='./logs'),
    tf.keras.callbacks.experimental.BackupAndRestore(backup_dir='./backup'),
    tf.keras.callbacks.LearningRateScheduler(step_decay)
]

steps_per_epoch = np.floor(len(os.listdir(x_train_dir)) / BATCH_SIZE)


model.fit(
    train_dist_dataset,
    steps_per_epoch=steps_per_epoch,
    epochs=1,
    callbacks=callbacks,
    # validation_data=val_dist_dataset,
    # validation_steps=len(os.listdir(x_valid_dir)),
)

task_type, task_id = (mirrored_strategy.cluster_resolver.task_type,
                      mirrored_strategy.cluster_resolver.task_id)

saved_model_dir = _get_saved_model_dir('saved_model_path', task_type, task_id)
model.save(os.path.join(saved_model_dir,'model.h5'))
if not _is_chief(task_type, task_id):
    tf.io.gfile.rmtree(os.path.dirname(saved_model_dir))
