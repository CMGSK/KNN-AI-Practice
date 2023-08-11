### DOCUMENTATION:
# Source of the dataset and its content: http://yann.lecun.com/exdb/mnist/

# IMAGE FILE:
# [offset] [type]          [value]          [description]
# 0000     32 bit integer  0x00000803(2051) magic number
# 0004     32 bit integer  60000            number of images
# 0008     32 bit integer  28               number of rows
# 0012     32 bit integer  28               number of columns
# 0016     unsigned byte   ??               pixel
# 0017     unsigned byte   ??               pixel
# ....     unsigned byte   ??               pixel

# Imports
import math
from PIL import Image
import numpy as np

# Define the directories and filenames we will use
TESTING_DIR = 'test/'
PREDICTION_DIR = 'pred/'
DATA_DIR = 'data/'
TEST_DATA_FILE = DATA_DIR + 't10k-images.idx3-ubyte'
TEST_LABELS_FILE = DATA_DIR + 't10k-labels.idx1-ubyte'
TRAIN_DATA_FILE = DATA_DIR + 'train-images.idx3-ubyte'
TRAIN_LABELS_FILE = DATA_DIR + 'train-labels.idx1-ubyte'


# Interpret bytes as an integer, since python doesnt support it
def byte_to_int(b):
    return int.from_bytes(b, 'big')  # Big/Little endian stands for bit readwise


# Read the images in file specified. If a limit is specified, real number gets overwritten
def read_images(file, img_limit=None):
    images = []
    # rb for read binary
    with open(file, 'rb') as f:
        # Read 4 bytes that stands for Magic number. Not interested in using it.
        _ = f.read(4)
        # Read the important bytes
        num_images = byte_to_int(f.read(4))
        if img_limit:
            num_images = img_limit
        num_rows = byte_to_int(f.read(4))
        num_columns = byte_to_int(f.read(4))
        # Mount them
        for image_idx in range(num_images):
            image = []
            for row_idx in range(num_rows):
                row = []
                for col_idx in range(num_columns):
                    pixel = f.read(1)
                    row.append(pixel)
                image.append(row)
            images.append(image)
        return images


# Reading images
def read_image(path):
    return np.asarray(Image.open(path).convert('L'))


# Writes the image at the specified path
def write_image(image, path):
    img = Image.fromarray(np.array(image), 'L')
    img.save(path)


def read_labels(file, lab_limit=None):
    labels = []
    # rb for read binary
    with open(file, 'rb') as f:
        # Read 4 bytes that stands for Magic number. Not interested in using it.
        _ = f.read(4)
        num_labels = byte_to_int(f.read(4))
        if lab_limit:
            num_labels = lab_limit
        for label_idx in range(num_labels):
            label = f.read(1)
            labels.append(label)
    return labels


# Flatten the image by creating a single list of pixels from every pixel in the image passed by
def pixel_rearrange(image):
    return [pixel for row in image for pixel in row]


# return a list of flattened images
def flatten_images(dataset):
    return [pixel_rearrange(image) for image in dataset]


# Get the sum of all pixel difference between the input image and the train image passed by.
def get_distances(a, b):
    return math.sqrt(sum([math.pow(byte_to_int(a_idx) - byte_to_int(b_idx), 2) for a_idx, b_idx in zip(a, b)]))


# Get the array of closest values to the image we're processing comparing to all the train data
def get_closest_values(img_train, image):
    return [get_distances(sample, image) for sample in img_train]


# Get the most frequent element in the prediction
def get_frequent(data):
    return max(data, key=data.count)
    # Not the maximum value, but the value that occurs most often.
    # .count() gives us a list of the count of every item in data like:
    # for a list with 5 zeros, each zero in the list will appear on it as a 5.


# AI Algorythm.
# K its out hyperparameter, which defines which K closest matches we compare our input with.
def knn_alg(img_train, lab_train, img_test, lab_test, k=3):
    label_prediction = []
    # Prediction for what the program think each image test is
    # for image in img_test:
    # We enumerate this for passing the input to the printf later and check if we guessed alright
    for input, image in enumerate(img_test):
        sorted_dist_idx = [
            pair[0]
            for pair in sorted(
                enumerate(get_closest_values(img_train, image)),
                key=lambda e: e[1]
            )
        ]
        # Accept predictions extracted from the train labels, based on the sorted distance up to K
        predictions = [
            lab_train[i]
            for i in sorted_dist_idx[:k]
        ]
        certainty = round((predictions.count(lab_test[input]) / k) * 99, 2)
        final_prediction = get_frequent(predictions)
        print(f'Input: {byte_to_int(lab_test[input])} | Guessing... {[byte_to_int(p) for p in predictions]}')
        print(f'Input: {byte_to_int(lab_test[input])} | Guessing {byte_to_int(final_prediction)} | Certainty {certainty}%')
        label_prediction.append(final_prediction)
    return label_prediction


def main():
    # Read the images
    img_train = read_images(TRAIN_DATA_FILE, 30000)
    lab_train = read_labels(TRAIN_LABELS_FILE, 30000)
    img_test = read_images(TEST_DATA_FILE, 20)
    lab_test = read_labels(TEST_LABELS_FILE, 20)

    PRED = True
    if PRED:
        img_test = [read_image(f'{PREDICTION_DIR}pic.png')]
    else:
        # Informative prints
        print(f'Number of train images: {len(img_train)}')
        print(f'Image resolution: {len(img_train[0])}x{len(img_train[0][0])}\n')
        print(f'Number of train labels: {len(lab_train)}\n')
        print('############\n')
        print(f'Number of test images: {len(img_test)}')
        print(f'Image resolution: {len(img_test[0])}x{len(img_test[0][0])}\n')
        print(f'Number of test labels: {len(lab_test)}\n')
        print(f'Flatten image resolution: {len(img_train[0])}x{len(img_train[0][0])}')
        print('############\n')
        # Just an image writer of out inputs at //test/
        for i, sample in enumerate(img_test):
            write_image(sample, f'{TESTING_DIR}Input {i}.png')

    # Image flattening
    img_train = flatten_images(img_train)
    img_test = flatten_images(img_test)

    # Algorythm call. Passing lab_test is not correct, we do it to check accuracy in debugging
    knn_alg(img_train, lab_train, img_test, lab_test, 7)

# Defines the entry point. Python convention.
if __name__ == '__main__':
    main()
