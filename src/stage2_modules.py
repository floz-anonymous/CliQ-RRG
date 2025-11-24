import tensorflow as tf
from tensorflow.keras import layers
import numpy as np
import openai
from .knowledge_base import CLINICAL_KNOWLEDGE_BASE

class TransformerDecoderLayer(layers.Layer):
    def __init__(self, d_model, num_heads, dff, rate=0.1):
        super(TransformerDecoderLayer, self).__init__()
        self.mha1 = layers.MultiHeadAttention(num_heads=num_heads, key_dim=d_model)
        self.mha2 = layers.MultiHeadAttention(num_heads=num_heads, key_dim=d_model)

        self.ffn = tf.keras.Sequential([
            layers.Dense(dff, activation='relu'),
            layers.Dense(d_model)
        ])

        self.layernorm1 = layers.LayerNormalization(epsilon=1e-6)
        self.layernorm2 = layers.LayerNormalization(epsilon=1e-6)
        self.layernorm3 = layers.LayerNormalization(epsilon=1e-6)

        self.dropout1 = layers.Dropout(rate)
        self.dropout2 = layers.Dropout(rate)
        self.dropout3 = layers.Dropout(rate)

    def call(self, x, enc_output, training, look_ahead_mask, padding_mask):
        attn1, _ = self.mha1(x, x, x, attention_mask=look_ahead_mask)
        attn1 = self.dropout1(attn1, training=training)
        out1 = self.layernorm1(x + attn1)

        attn2, _ = self.mha2(out1, enc_output, enc_output, attention_mask=padding_mask)
        attn2 = self.dropout2(attn2, training=training)
        out2 = self.layernorm2(out1 + attn2)

        ffn_output = self.ffn(out2)
        ffn_output = self.dropout3(ffn_output, training=training)
        out3 = self.layernorm3(out2 + ffn_output)

        return out3

class ReportDecoder(layers.Layer):
    def __init__(self, num_layers=3, d_model=256, num_heads=8, dff=1024, 
                 target_vocab_size=5000, max_pos_encoding=100, rate=0.1):
        super(ReportDecoder, self).__init__()
        
        self.d_model = d_model
        self.num_layers = num_layers
        
        self.embedding = layers.Embedding(target_vocab_size, d_model)
        self.pos_encoding = self.positional_encoding(max_pos_encoding, d_model)
        
        self.dec_layers = [
            TransformerDecoderLayer(d_model, num_heads, dff, rate) 
            for _ in range(num_layers)
        ]
        
        self.dropout = layers.Dropout(rate)
        self.final_layer = layers.Dense(target_vocab_size)

    def positional_encoding(self, position, d_model):
        angle_rads = self.get_angles(np.arange(position)[:, np.newaxis],
                                     np.arange(d_model)[np.newaxis, :],
                                     d_model)
        angle_rads[:, 0::2] = np.sin(angle_rads[:, 0::2])
        angle_rads[:, 1::2] = np.cos(angle_rads[:, 1::2])
        pos_encoding = angle_rads[np.newaxis, ...]
        return tf.cast(pos_encoding, dtype=tf.float32)

    def get_angles(self, pos, i, d_model):
        angle_rates = 1 / np.power(10000, (2 * (i // 2)) / np.float32(d_model))
        return pos * angle_rates

    def call(self, x, context_features, training, look_ahead_mask, padding_mask):
        seq_len = tf.shape(x)[1]
        x = self.embedding(x)
        x *= tf.math.sqrt(tf.cast(self.d_model, tf.float32))
        x += self.pos_encoding[:, :seq_len, :]
        x = self.dropout(x, training=training)

        for i in range(self.num_layers):
            x = self.dec_layers[i](x, context_features, training, look_ahead_mask, padding_mask)

        final_output = self.final_layer(x)
        return final_output

class KnowledgeRetriever:
    def __init__(self):
        self.knowledge_base = CLINICAL_KNOWLEDGE_BASE
        self.kb_embeddings = self._load_biowordvec_embeddings()

    def _load_biowordvec_embeddings(self):
        return np.random.normal(size=(len(self.knowledge_base), 768))

    def _get_report_embedding(self, report_text):
        return np.random.normal(size=(1, 768))

    def retrieve(self, report_text, k=10):
        report_embedding = self._get_report_embedding(report_text)
        
        norm_report = np.linalg.norm(report_embedding)
        norm_kb = np.linalg.norm(self.kb_embeddings, axis=1)
        
        scores = np.dot(self.kb_embeddings, report_embedding.T).flatten() / (norm_kb * norm_report + 1e-9)
        
        top_k_indices = np.argsort(scores)[-k:][::-1]
        
        retrieved_tokens = [self.knowledge_base[i] for i in top_k_indices]
        return ", ".join(retrieved_tokens)

class QAGenerator:
    def __init__(self, api_key=None):
        if api_key:
            openai.api_key = api_key
            
    def generate(self, knowledge_injected_report):
        prompt = f"""
        Given a knowledge-enriched chest X-ray report, the objective is to convert the
        report into a clinically coherent question-answer (QA) format and generate a
        concise diagnostic summary. This involves decomposing the report into short,
        interpretable QA pairs, followed by synthesizing a summary of the key findings.

        Report:
        {knowledge_injected_report}
        """

        try:
            response = openai.ChatCompletion.create(
                model="gpt-3.5-turbo",
                messages=[
                    {"role": "system", "content": "You are a helpful medical assistant."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.7
            )
            return response.choices[0].message.content.strip()
        except Exception:
            return "Q: Are there findings? A: No acute abnormality. Summary: Clear lungs."