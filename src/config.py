import os

DIR_ROOT = os.path.abspath('./')
DIR_COLOR_IMAGES = os.path.join(DIR_ROOT, 'dataset/color/')
DIR_GRAY_IMAGES = os.path.join(DIR_ROOT, 'dataset/gray/')

# ------------------------------------------------------------ #

IMAGE_SIZE = 224
BATCH_SIZE = 10
NUM_EPOCHS = 10
GRADIENT_PENALTY_WEIGHT = 10