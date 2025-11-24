import tensorflow as tf
from tensorflow import keras
from .modules import VisualEncoder, TextualEncoder, PriorGuidedAttention, DiseaseClassifier
from .losses import DiseaseAwareContrastiveLoss

class CliQRRG_Stage1(keras.Model):
    def __init__(self, embed_dim=768, num_classes=13, temperature=0.07, **kwargs):
        super(CliQRRG_Stage1, self).__init__(**kwargs)
        
        self.visual_encoder = VisualEncoder(embed_dim=embed_dim)
        self.pram = PriorGuidedAttention(embed_dim=embed_dim)
        self.textual_encoder = TextualEncoder(embed_dim=embed_dim)
        self.disease_classifier = DiseaseClassifier(num_classes=num_classes, embed_dim=embed_dim)
        
        self.contrastive_loss_fn = DiseaseAwareContrastiveLoss(temperature=temperature)
        self.cls_loss_fn = keras.losses.CategoricalCrossentropy()

    def call(self, inputs, training=False):
        current_image = inputs['current_image']
        prior_images = inputs['prior_images']
        input_ids = inputs['input_ids']
        attention_mask = inputs['attention_mask']
        
        v_curr_embed = self.visual_encoder(current_image)
        
        b_size = tf.shape(prior_images)[0]
        num_priors = tf.shape(prior_images)[1]
        
        prior_images_reshaped = tf.reshape(prior_images, [-1, tf.shape(prior_images)[2], tf.shape(prior_images)[3], tf.shape(prior_images)[4]])
        v_priors_flat = self.visual_encoder(prior_images_reshaped)
        v_priors_embed = tf.reshape(v_priors_flat, [b_size, num_priors, -1])
        
        v_ch = self.pram(v_curr_embed, v_priors_embed)
        
        t_ch = self.textual_encoder(input_ids, attention_mask)
        
        pred_probs, pred_logits = self.disease_classifier(v_ch)
        
        return {
            "v_ch": v_ch,
            "t_ch": t_ch,
            "pred_probs": pred_probs,
            "pred_logits": pred_logits
        }

    def train_step(self, data):
        inputs, disease_labels_gt = data
        
        with tf.GradientTape() as tape:
            outputs = self(inputs, training=True)
            
            v_ch = outputs['v_ch']
            t_ch = outputs['t_ch']
            pred_probs = outputs['pred_probs']
            
            loss_vtc = self.contrastive_loss_fn(v_ch, t_ch, disease_labels_gt)
            loss_ce = self.cls_loss_fn(disease_labels_gt, pred_probs)
            
            total_loss = loss_vtc + loss_ce
            
        gradients = tape.gradient(total_loss, self.trainable_variables)
        self.optimizer.apply_gradients(zip(gradients, self.trainable_variables))
        
        return {
            "loss": total_loss,
            "vtc_loss": loss_vtc,
            "ce_loss": loss_ce
        }