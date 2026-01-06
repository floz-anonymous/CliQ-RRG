import tensorflow as tf
from tensorflow.keras import layers
from transformers import TFAutoModel, TFCLIPVisionModel

class VisualEncoder(layers.Layer):
    def __init__(self, model_name="microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224", embed_dim=768, **kwargs):
        super(VisualEncoder, self).__init__(**kwargs)
        self.embed_dim = embed_dim
        # Load the BioMedCLIP Vision Transformer (ViT-B/16)
        # Note: from_pt=True requires PyTorch to be installed to convert weights
        self.backbone = TFCLIPVisionModel.from_pretrained(model_name, from_pt=True)
        
        # Keep projection to map to the shared embedding space dimension
        self.projection = layers.Dense(embed_dim)

    def call(self, images):
        # Transformers expect NCHW (Batch, Channels, Height, Width)
        # TensorFlow images are usually NHWC (Batch, Height, Width, Channels)
        images = tf.transpose(images, perm=[0, 3, 1, 2])
        
        # Pass pixel_values to the ViT backbone
        outputs = self.backbone(pixel_values=images)
        
        # Use pooler_output (CLS token features) as the image representation
        features = outputs.pooler_output
        return self.projection(features)

class TextualEncoder(layers.Layer):
    def __init__(self, model_name="microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract-fulltext", embed_dim=768, **kwargs):
        super(TextualEncoder, self).__init__(**kwargs)
        self.embed_dim = embed_dim
        self.bert = TFAutoModel.from_pretrained(model_name, from_pt=True)
        self.projection = layers.Dense(embed_dim)

    def call(self, input_ids, attention_mask):
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        cls_token = outputs.last_hidden_state[:, 0, :]
        return self.projection(cls_token)

class PriorGuidedAttention(layers.Layer):
    def __init__(self, embed_dim, **kwargs):
        super(PriorGuidedAttention, self).__init__(**kwargs)
        self.embed_dim = embed_dim
        self.scale = tf.sqrt(tf.cast(embed_dim, tf.float32))

    def call(self, v_current, v_priors):
        Q = tf.expand_dims(v_current, axis=1)
        K = v_priors
        V = v_priors
        scores = tf.matmul(Q, K, transpose_b=True) / self.scale
        attn_weights = tf.nn.softmax(scores, axis=-1)
        context = tf.matmul(attn_weights, V)
        context = tf.squeeze(context, axis=1)
        return v_current + context

class DiseaseClassifier(layers.Layer):
    def __init__(self, num_classes=13, embed_dim=768, **kwargs):
        super(DiseaseClassifier, self).__init__(**kwargs)
        self.num_classes = num_classes
        self.embed_dim = embed_dim
        self.scale = tf.sqrt(tf.cast(embed_dim, tf.float32))
        self.phi = self.add_weight(
            shape=(num_classes, embed_dim),
            initializer="glorot_uniform",
            trainable=True,
            name="disease_embedding_phi"
        )

    def call(self, visual_features):
        logits = tf.matmul(visual_features, self.phi, transpose_b=True) / self.scale
        probs = tf.nn.softmax(logits, axis=-1)
        return probs, logits
