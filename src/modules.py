import tensorflow as tf
from tensorflow.keras import layers
from transformers import TFAutoModel

class VisualEncoder(layers.Layer):
    def __init__(self, embed_dim=768, **kwargs):
        super(VisualEncoder, self).__init__(**kwargs)
        self.embed_dim = embed_dim
        self.backbone = tf.keras.applications.ResNet50(
            include_top=False,
            weights='imagenet',
            pooling='avg'
        )
        self.projection = layers.Dense(embed_dim)

    def call(self, images):
        features = self.backbone(images)
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
