import tensorflow as tf
from tensorflow.keras import layers

class DiseaseAwareContrastiveLoss(layers.Layer):
    def __init__(self, temperature=0.07, **kwargs):
        super(DiseaseAwareContrastiveLoss, self).__init__(**kwargs)
        self.temperature = temperature

    def call(self, v_features, t_features, disease_labels):
        v_norm = tf.math.l2_normalize(v_features, axis=1)
        t_norm = tf.math.l2_normalize(t_features, axis=1)
        
        logits_vt = tf.matmul(v_norm, t_norm, transpose_b=True) / self.temperature
        logits_tv = tf.matmul(t_norm, v_norm, transpose_b=True) / self.temperature
        
        batch_size = tf.shape(v_features)[0]
        
        labels = tf.range(batch_size)
        loss_vt = tf.keras.losses.sparse_categorical_crossentropy(labels, logits_vt, from_logits=True)
        loss_tv = tf.keras.losses.sparse_categorical_crossentropy(labels, logits_tv, from_logits=True)
        
        return (loss_vt + loss_tv) / 2.0