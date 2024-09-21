import numpy as np
from keras.utils import Sequence
import tensorflow as tf
import cv2

import csv
import os, random
from pathlib import Path
from glob import glob
class DataGeneratorGAN(Sequence):
    def __init__(self,frames_path_dict, num_classes, batch_size=32, to_fit=True, shuffle=True, **kwargs):
        super().__init__(**kwargs)
        self.frames_path_dict = frames_path_dict
        self.to_fit = to_fit
        self.num_classes = num_classes
        self.batch_size = batch_size
        self.list_IDS = list(range(0, len(frames_path_dict)))
        self.shuffle = shuffle
        self.on_epoch_end()   

    def __len__(self):
        return int(np.floor(len(self.frames_path_dict)) / self.batch_size)     

    
    def __getitem__(self, index):
        """Generate one batch of data
        :param index: index of the batch
        :return: X and y when fitting. X only when predicting
        """
        # Generate indexes of the batch
        indexes = self.indexes[index * self.batch_size:(index + 1) * self.batch_size]

        # Find list of IDs
        list_IDs_temp = [self.list_IDS[k] for k in indexes]
        
        # Generate data
        frames_batch, labels_batch = self.__generate_frames_ds__(list_IDs_temp)

        if self.to_fit == True:
            return frames_batch, labels_batch
        else:
            return frames_batch


    def on_epoch_end(self):
        # Updates indexes after each epoch
        self.indexes = np.arange(len(self.list_IDS))
        if self.shuffle == True:
            np.random.shuffle(self.indexes)

    def __generate_frames_ds__(self, list_IDs_temp):
        frame_ds = np.empty((self.batch_size, 1920//3, 1080//3, 3), dtype=np.float32)
        # frame_ds = np.empty((self.batch_size, 1920 // 2, 1080 // 2, 3), dtype=np.float32)
        labels_ds = np.empty((self.batch_size, self.num_classes), dtype=np.float32)
        for i, id in enumerate(list_IDs_temp):
            key = "item_ID"
            val = id
            item = next((d for d in self.frames_path_dict if d.get(key) == val), None)
            
            frame = self.__get_image__(item["patch_path"])
            label = item["class_label"]
            frame_ds[i, ...] = frame
            labels_ds[i, ...] = label
            
        return frame_ds, labels_ds

    def __get_image__(self, image_path):
        img = cv2.imread(image_path)
        # if dimensions are 1920x1080 switch to 1080x1920
        # print(img.shape)

        if img.shape[0] == 1080 and img.shape[1] == 1920:
            img = np.transpose(img, (1, 0, 2))
        height, width = img.shape[:2]
        img = cv2.resize(img, (width//3, height//3))
        
        # img = apply_cfa(img)
        # Normalize the pixel values
        img = tf.cast(img, tf.float32)
        img = (img - 127.5) / 127.5
        return img
    
class DataSetGeneratorGAN:
    def __init__(self, input_dir_patchs: Path =None, classes: list = None, excluded_devices: list = None): 
        self.data_dir_patchs = Path(input_dir_patchs)        
        self.device_types = np.array([item.name for item in self.data_dir_patchs.glob('*') if not item.name.startswith('.')])

        self.train_image_count = len(list(self.data_dir_patchs.glob(f'**/Training/**/*.jpg')))
        self.test_image_count = len(list(self.data_dir_patchs.glob(f'**/Testing/**/*.jpg')))
        self.val_image_count = len(list(self.data_dir_patchs.glob(f'**/Validation/**/*.jpg')))


        self.exclude_devices = excluded_devices if excluded_devices is not None else []
        self.class_names = self.get_classes(classes)
        print(self.class_names)


    def get_classes(self, classes):
        if classes is not None:
            return classes
        else:
            class_names = sorted(self.device_types)
            all_classes = np.array(class_names) 
            final_classes = []
            for c in all_classes:
                if c not in self.exclude_devices:
                    final_classes.append(c)
            return np.array(final_classes)  

    def get_class_names(self):
        return self.class_names

    def device_count(self):
        return len(self.class_names)

    def listdir_nonhidden(path):
        return [f for f in os.listdir(path) if not f.startswith('.')]

    def determine_label(self, file_path):
        classes = self.get_class_names()
        label_vector_lenght = self.device_count()
        label = np.zeros((label_vector_lenght,), dtype=int)
        classes.sort()
        for i, class_name in enumerate(classes):
            if class_name in file_path:
                label[i] = 1
        return label
    
    def create_dataset(self, type='Training') -> list:
        input_path_file_names = np.array(glob(str(self.data_dir_patchs) + f"/**/{type}/**/*.jpg", recursive = True))
        input_patchs_file_names_temp = []
        for file in input_path_file_names:
            input_patchs_file_names_temp.append(file)
        input_patchs_file_names_final = np.array(input_patchs_file_names_temp)

        labeled_dictionary = list()
        random.shuffle(input_patchs_file_names_final)

        for i, file_path in enumerate(input_patchs_file_names_final):
            class_label = self.determine_label(file_path)
            ds_row = {"item_ID": i, "patch_path": file_path, "class_label": class_label}                        
            labeled_dictionary.append(ds_row)

        return labeled_dictionary