import os

DIR_ROOT = os.path.abspath('./')
DIR_IMAGES = os.path.join(DIR_ROOT, 'grayscale-dataset-data/dataset/')

# ------------------------------------------------------------ #

IMAGE_SIZE = 224
BATCH_SIZE = 10
NUM_EPOCHS = 10
GRADIENT_PENALTY_WEIGHT = 10