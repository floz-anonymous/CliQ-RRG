import os
import argparse
import tensorflow as tf
from transformers import AutoTokenizer
from src.model_stage1 import CliQRRG_Stage1
from src.dataloader import RadiologyDataset

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default='mimic', choices=['mimic', 'iu'], help='Dataset name')
    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--epochs', type=int, default=30)
    parser.add_argument('--lr', type=float, default=3e-5)
    parser.add_argument('--embed_dim', type=int, default=768)
    parser.add_argument('--num_classes', type=int, default=13)
    parser.add_argument('--temp', type=float, default=0.07)
    parser.add_argument('--data_dir', type=str, required=True, help="Path to image directory")
    parser.add_argument('--csv_file', type=str, required=True, help="Path to train CSV")
    return parser.parse_args()

def main():
    args = parse_args()
    
    strategy = tf.distribute.MirroredStrategy()
    print(f"Training on {strategy.num_replicas_in_sync} GPUs")

    # Initialize Tokenizer
    tokenizer = AutoTokenizer.from_pretrained("microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract-fulltext")

    # Create Dataset Pipeline
    data_loader = RadiologyDataset(
        data_dir=args.data_dir,
        csv_file=args.csv_file,
        tokenizer=tokenizer,
        dataset_name=args.dataset,
        batch_size=args.batch_size
    )
    train_dataset = data_loader.get_dataset()

    with strategy.scope():
        model = CliQRRG_Stage1(
            embed_dim=args.embed_dim,
            num_classes=args.num_classes,
            temperature=args.temp
        )
        
        optimizer = tf.keras.optimizers.AdamW(learning_rate=args.lr, weight_decay=0.01)
        model.compile(optimizer=optimizer)

    checkpoint_dir = f'./checkpoints/{args.dataset}_stage1'
    if not os.path.exists(checkpoint_dir):
        os.makedirs(checkpoint_dir)
        
    callbacks = [
        tf.keras.callbacks.ModelCheckpoint(
            filepath=os.path.join(checkpoint_dir, 'ckpt_{epoch}'),
            save_weights_only=True,
            save_freq='epoch'
        ),
        tf.keras.callbacks.TensorBoard(log_dir=f'./logs/{args.dataset}_stage1')
    ]

    print(f"Starting Training Stage 1 on {args.dataset}...")
    model.fit(
        train_dataset, 
        epochs=args.epochs,
        callbacks=callbacks
    )

if __name__ == "__main__":
    main()