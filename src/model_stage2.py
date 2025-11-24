import tensorflow as tf
from tensorflow import keras
from .modules import VisualEncoder, PriorGuidedAttention, DiseaseClassifier
from .stage2_modules import ReportDecoder

class CliQRRG_Stage2(keras.Model):
    def __init__(self, stage1_checkpoint=None, vocab_size=5000, **kwargs):
        super(CliQRRG_Stage2, self).__init__(**kwargs)
        
        self.visual_encoder = VisualEncoder()
        self.pram = PriorGuidedAttention(embed_dim=768)
        self.disease_classifier = DiseaseClassifier()
        
        self.feature_projection = keras.layers.Dense(256)
        
        self.decoder = ReportDecoder(target_vocab_size=vocab_size, d_model=256)
        
        self.loss_fn = keras.losses.SparseCategoricalCrossentropy(from_logits=True, reduction='none')

    def call(self, inputs, training=False):
        current_image = inputs['current_image']
        prior_images = inputs['prior_images']
        target_seq = inputs['target_seq']
        
        v_curr = self.visual_encoder(current_image)
        
        b_size = tf.shape(prior_images)[0]
        num_priors = tf.shape(prior_images)[1]
        prior_flat = tf.reshape(prior_images, [-1, tf.shape(prior_images)[2], tf.shape(prior_images)[3], tf.shape(prior_images)[4]])
        v_prior_flat = self.visual_encoder(prior_flat)
        v_priors = tf.reshape(v_prior_flat, [b_size, num_priors, -1])
        
        v_ch = self.pram(v_curr, v_priors)
        
        _, l_logits = self.disease_classifier(v_ch)
        
        context_raw = tf.concat([v_ch, l_logits], axis=-1) 
        
        context = self.feature_projection(context_raw)
        context = tf.expand_dims(context, 1) 
        
        look_ahead_mask = self.create_look_ahead_mask(tf.shape(target_seq)[1])
        padding_mask = self.create_padding_mask(context)
        
        predictions = self.decoder(
            target_seq, 
            context, 
            training, 
            look_ahead_mask, 
            padding_mask
        )
        
        return predictions

    def create_look_ahead_mask(self, size):
        mask = 1 - tf.linalg.band_part(tf.ones((size, size)), -1, 0)
        return mask
        
    def create_padding_mask(self, seq):
        return tf.cast(tf.math.equal(seq, 0), tf.float32)[:, tf.newaxis, tf.newaxis, :]

    def train_step(self, data):
        inputs, target_tokens = data
        
        tar_inp = target_tokens[:, :-1]
        tar_real = target_tokens[:, 1:]
        
        inputs['target_seq'] = tar_inp
        
        with tf.GradientTape() as tape:
            predictions = self(inputs, training=True)
            loss = self.loss_function(tar_real, predictions)
            
        gradients = tape.gradient(loss, self.trainable_variables)
        self.optimizer.apply_gradients(zip(gradients, self.trainable_variables))
        
        return {'loss': loss}

    def loss_function(self, real, pred):
        mask = tf.math.logical_not(tf.math.equal(real, 0))
        loss_ = self.loss_fn(real, pred)
        mask = tf.cast(mask, dtype=loss_.dtype)
        loss_ *= mask
        return tf.reduce_sum(loss_) / tf.reduce_sum(mask)