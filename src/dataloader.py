import tensorflow as tf
import pandas as pd
import numpy as np
import os

class RadiologyDataset:
    def __init__(self, data_dir, csv_file, tokenizer, dataset_name='mimic', batch_size=16, img_size=(224, 224), max_len=64):
        self.data_dir = data_dir
        self.df = pd.read_csv(csv_file)
        self.tokenizer = tokenizer
        self.dataset_name = dataset_name.lower()
        self.batch_size = batch_size
        self.img_size = img_size
        self.max_len = max_len

    def _process_image(self, img_path):
        img_raw = tf.io.read_file(img_path)
        if self.dataset_name == 'iu':
             img = tf.io.decode_png(img_raw, channels=3)
        else:
             img = tf.io.decode_jpeg(img_raw, channels=3)
             
        img = tf.image.resize(img, self.img_size)
        img = tf.image.per_image_standardization(img)
        return img

    def _process_text(self, text):
        tokens = self.tokenizer(
            str(text), 
            padding='max_length', 
            truncation=True, 
            max_length=self.max_len, 
            return_tensors='tf'
        )
        return tokens['input_ids'][0], tokens['attention_mask'][0]

    def _parse_function(self, current_path, prior_path, report, labels):
        curr_img = self._process_image(current_path)
        
        if prior_path == b'None' or prior_path == b'':
             prior_imgs = tf.zeros_like(tf.expand_dims(curr_img, 0))
        else:
             prior_img = self._process_image(prior_path)
             prior_imgs = tf.expand_dims(prior_img, 0)
        
        input_ids, attn_mask = tf.py_function(
            func=self._process_text, 
            inp=[report], 
            Tout=[tf.int32, tf.int32]
        )
        
        inputs = {
            'current_image': curr_img,
            'prior_images': prior_imgs,
            'input_ids': input_ids,
            'attention_mask': attn_mask
        }
        return inputs, labels

    def get_dataset(self):
        if self.dataset_name == 'mimic':
            img_col = 'dicom_id'
            prior_col = 'prior_dicom_id' 
            text_col = 'text'
        elif self.dataset_name == 'iu':
            img_col = 'image_path'
            prior_col = 'prior_image_path' 
            text_col = 'report'
        else:
            raise ValueError("Unknown dataset name. Use 'mimic' or 'iu'.")

        current_paths = [os.path.join(self.data_dir, str(x)) for x in self.df[img_col].values]
        
        if prior_col in self.df.columns:
             prior_paths = [os.path.join(self.data_dir, str(x)) if str(x) != 'nan' else 'None' for x in self.df[prior_col].values]
        else:
             prior_paths = ['None'] * len(self.df)

        reports = self.df[text_col].values
        
        if 'labels' in self.df.columns:
             labels = np.array([np.fromstring(x, sep=' ') for x in self.df['labels'].values], dtype=np.float32)
        else:
             labels = np.zeros((len(self.df), 13), dtype=np.float32)

        dataset = tf.data.Dataset.from_tensor_slices((current_paths, prior_paths, reports, labels))
        dataset = dataset.map(lambda c, p, r, l: tf.py_function(self._parse_function, [c, p, r, l], [dict, tf.float32]))
        
        def set_shapes(inputs, labels):
            inputs['current_image'].set_shape((self.img_size[0], self.img_size[1], 3))
            inputs['prior_images'].set_shape((None, self.img_size[0], self.img_size[1], 3))
            inputs['input_ids'].set_shape((self.max_len,))
            inputs['attention_mask'].set_shape((self.max_len,))
            labels.set_shape((13,))
            return inputs, labels

        dataset = dataset.map(set_shapes)
        dataset = dataset.shuffle(1000).batch(self.batch_size).prefetch(tf.data.AUTOTUNE)
        return dataset