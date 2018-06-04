import random
from casia_tfrecord_helper import _dataset_exists, _get_filenames_and_classes, write_label_file, _convert_dataset

TFRECORD_FILENAME = 'tfcasia'
NUM_SHARDS=1
DATASET_DIR='/home/wangrc/dataset/'
FOLDER_NAME='casia_mtcnn'
VALIDATION_SET_SPLIT_RATIO=0.1
RANDOM_SEED=1
def main():

    # If the TFRecord files already exist in the directory, then exit without creating the files again
    if _dataset_exists(dataset_dir=DATASET_DIR, _NUM_SHARDS=NUM_SHARDS,
                       output_filename=TFRECORD_FILENAME):
        print('Dataset files already exist. Exiting without re-creating them.')
        return None

    # Get a list of photo_filenames like ['123.jpg', '456.jpg'...] and a list of sorted class names from parsing the subdirectories.
    photo_filenames, class_names = _get_filenames_and_classes(DATASET_DIR,FOLDER_NAME)
    print ('total classes: ')
    print (len(class_names))
    # Refer each of the class name to a specific integer number for predictions later
    class_names_to_ids = dict(zip(class_names, range(len(class_names))))

    # Find the number of validation examples we need
    num_validation = int(VALIDATION_SET_SPLIT_RATIO * len(photo_filenames))

    # Divide the training datasets into train and test:
    random.seed(RANDOM_SEED)
    random.shuffle(photo_filenames)
    training_filenames = photo_filenames[num_validation:]
    validation_filenames = photo_filenames[:num_validation]

    # Convert the training and validation sets.
    _convert_dataset('train', training_filenames, class_names_to_ids,
                     dataset_dir=DATASET_DIR, tfrecord_filename=TFRECORD_FILENAME,
                     _NUM_SHARDS=NUM_SHARDS)
    _convert_dataset('validation', validation_filenames, class_names_to_ids,
                     dataset_dir=DATASET_DIR, tfrecord_filename=TFRECORD_FILENAME,
                     _NUM_SHARDS=NUM_SHARDS)

    # Finally, write the labels file:
    labels_to_class_names = dict(zip(range(len(class_names)), class_names))
    write_label_file(labels_to_class_names, DATASET_DIR)

    print('\nFinished converting the %s dataset!' % TFRECORD_FILENAME)


if __name__ == "__main__":
    main()