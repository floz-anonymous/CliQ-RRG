import argparse
import tensorflow as tf
from src.model_stage2 import CliQRRG_Stage2
from src.stage2_modules import KnowledgeRetriever, QAGenerator

def run_inference(args):
    model = CliQRRG_Stage2(vocab_size=args.vocab_size)
    dummy_input = {
        'current_image': tf.random.normal((1, 224, 224, 3)),
        'prior_images': tf.random.normal((1, 2, 224, 224, 3)),
        'target_seq': tf.zeros((1, 1), dtype=tf.int32)
    }
    intermediate_report = "Lungs are well expanded and clear. No consolidation, effusion, or pneumothorax."
    retriever = KnowledgeRetriever()
    knowledge_tokens = retriever.retrieve(intermediate_report, k=10)
    knowledge_injected_report = f"{intermediate_report} [SEP] {knowledge_tokens}"
    qa_gen = QAGenerator(api_key=args.openai_api_key)
    final_report = qa_gen.generate(knowledge_injected_report)
    print("-" * 40)
    print("FINAL QA REPORT")
    print("-" * 40)
    print(final_report)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--vocab_size', type=int, default=5000)
    parser.add_argument('--checkpoint_path', type=str, default='./checkpoints/stage2')
    parser.add_argument('--openai_api_key', type=str, default='')
    args = parser.parse_args()
    run_inference(args)
