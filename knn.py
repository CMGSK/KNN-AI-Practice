### DOCUMENTATION:
# Source of the dataset and its content: http://yann.lecun.com/exdb/mnist/

# IMAGE FILE:
# [offset] [type]          [value]          [description]
# 0000     32 bit integer  0x00000803(2051) magic number
# 0004     32 bit integer  60000            number of images
# 0008     32 bit integer  28               number of rows
# 0012     32 bit integer  28               number of columns
# 0016     unsigned byte   ??               pixel
# ....     unsigned byte   ??               pixel

# Define the directories and filenames we will use
DATA_DIR = 'data/'
TEST_DATA_FILE = DATA_DIR + 't10k-images.idx3-ubyte'
TEST_LABELS_FILE = DATA_DIR + 't10k-labels.idx1-ubyte'
TRAIN_DATA_FILE = DATA_DIR + 'train-images.idx3-ubyte'
TRAIN_LABELS_FILE = DATA_DIR + 'train-labels.idx1-ubyte'

# To interpret bytes as an integer, since python doesnt support it
def byte_to_int (b):
    return int.from_bytes(b, 'big') # Big/Little endian stands for bit readwise


# Read the images in file specified
def read_images(file):
    images = []
    # rb for read binary
    with open(file, 'rb') as f:
        _ = f.read(4) # Reading 4 bytes that stands for Magic number. Not interested in using it.
        num_images = byte_to_int(f.read(4))
        num_rows = byte_to_int(f.read(4))
        num_columns = byte_to_int(f.read(4))
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



def main():
    img_train = read_images(TRAIN_DATA_FILE)
    # lab_train = read_images(TRAIN_LABELS_FILE)
    # img_test = read_images(TEST_DATA_FILE)
    # lab_test = read_images(TEST_LABELS_FILE)
    print(len(img_train))

# Allows the application to run as a script but not as a module. Python convention.
if __name__ == '__main__':
    main()
