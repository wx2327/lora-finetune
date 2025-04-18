# AG News Classification with RoBERTa + LoRA

This project fine-tunes a RoBERTa-base model using LoRA (Low-Rank Adaptation) for efficient text classification on the AG News dataset. The workflow covers data preprocessing, model training, evaluation, and inference.

## Data Format Example

Each sample in the AG News dataset contains a news text and a label (0-3):

```
{
  "text": "Wall St. Bears Claw Back Into the Black (Reuters) Reuters - Short-sellers, Wall Street's dwindling band of ultra-cynics, are seeing green again.",
  "label": 2
}
```

After tokenization, the format is:
```
{
  "input_ids": [0, 31414, 232, ...],
  "attention_mask": [1, 1, 1, ...],
  "labels": 2
}
```

Unlabeled test data only contains the `text` field. Inference results are saved as CSV:

| ID  | Label |
|-----|-------|
| 0   | 2     |
| 1   | 0     |
| ... | ...   |

## Parameter Settings

- Model: `roberta-base`
- LoRA rank (`r`): 7
- LoRA alpha: 16
- LoRA dropout: 0.1
- Target modules: ['query', 'key', 'value']
- Batch size (train/eval): 32/64
- Learning rate: 5e-5
- Epochs: 5
- Early stopping patience: 5
- Max trainable parameters: < 1M

## Final Kaggle Result

**Public Score:** 0.84700

## How to Run

1. Clone this repository and enter the project directory.
2. Open `Starter_Notebook.ipynb` with Jupyter Notebook or VS Code.
3. Run the first cell to install all dependencies (internet required for first run).
4. Execute each cell in order:
   - Data will be automatically loaded and preprocessed.
   - The model will be configured and fine-tuned with LoRA.
   - Training progress and evaluation metrics will be displayed.
   - The best model will be saved to `best_model/`.
   - Inference on the test set will generate `results/inference_output.csv` for Kaggle submission.
5. For custom inference, modify the relevant cell to input your own text.

**Note:**
- GPU is strongly recommended for training.
- The notebook is self-contained and can be run from top to bottom.

## Directory Structure

- `Starter_Notebook.ipynb`: Main notebook with all code and comments
- `best_model/`: Directory for the best model checkpoint
- `results/inference_output.csv`: Inference results for submission
- `test_unlabelled.pkl`: Unlabeled test set

## References

- [Hugging Face Transformers](https://huggingface.co/docs/transformers)
- [PEFT (LoRA)](https://github.com/huggingface/peft)
- [AG News Dataset](https://huggingface.co/datasets/ag_news)
