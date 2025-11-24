# CliQ-RRG: Clinical-Knowledge Guided Disease-aware Visual-Textual Alignment

This repository contains the official implementation of **CliQ-RRG**, a unified two-stage framework for generating structured, QA-style radiology reports. This work leverages multi-view chest X-rays (CXRs), prior-guided attention, and clinical knowledge injection to improve diagnostic accuracy and interpretability.

## Project Structure
```
├── src/
│   ├── modules.py         # VisualEncoder, PrAM, TextualEncoder, DiseaseClassifier
│   ├── losses.py          # Disease-aware Contrastive Loss
│   ├── model_stage1.py    # Visual-Textual Alignment
│   ├── model_stage2.py    # Decoder & Knowledge Injection
│   ├── stage2_modules.py  # ReportDecoder, KnowledgeRetriever, QAGenerator
│   ├── knowledge_base.py  # Clinical attributes
│   └── dataloader.py      # Data loading pipeline
├── train_stage1.py        # Training script for Stage 1
├── inference.py           # Inference pipeline for generating QA reports
└── requirements.txt       # Python dependencies
```

## Prerequisites
- Python 3.8+
- TensorFlow 2.10+
- NVIDIA GPU (Recommended for training)

## Installation
```bash
pip install -r requirements.txt
```

## Data Preparation
The `src/dataloader.py` script is designed to handle **MIMIC-CXR** and **IU X-Ray** datasets. Ensure your data is organized with a CSV file containing the following columns:

- **MIMIC-CXR**: `dicom_id`, `prior_dicom_id` (optional), `text, labels`

- **IU X-Ray**: `image_path`, `report`, `labels`

Update the `data_dir` argument in the training scripts to point to your local image directory.

## Usage

**1. Stage 1: Disease-aware Visual-Textual Alignment**

Train the alignment module. This stage uses the components in `src/modules.py` and optimizes the custom contrastive loss defined in `src/losses.py`.

**To train on MIMIC-CXR:**

```bash
python train_stage1.py \
  --dataset mimic \
  --data_dir /path/to/mimic/images \
  --csv_file /path/to/mimic_train.csv \
  --batch_size 16 \
  --epochs 30 \
  --lr 3e-5
```

**To train on IU X-Ray:**

```bash
python train_stage1.py \
  --dataset iu \
  --data_dir /path/to/iu/images \
  --csv_file /path/to/iu_train.csv \
  --batch_size 16
```

This process initializes the `CliQRRG_Stage1` model from `src/model_stage1.py` and saves model checkpoints to the `./checkpoints/` directory.

**2. Stage 2: QA-Style Report Generation**

Run the full inference pipeline using `inference.py`. This script performs the following steps:

Loads the trained Stage 2 architecture (`src/model_stage2.py`).

Generates an intermediate narrative report using the `ReportDecoder` found in `src/stage2_modules.py`.

Retrieves relevant clinical context from `src/knowledge_base.py` via the `KnowledgeRetriever`.

Synthesizes the final QA output using the `QAGenerator` (requires OpenAI API key).

```bash
python inference.py \
  --vocab_size 5000 \
  --checkpoint_path ./checkpoints/stage2 \
  --openai_api_key "YOUR_API_KEY"
  ```

## License

This project is released under the **MIT License**.
