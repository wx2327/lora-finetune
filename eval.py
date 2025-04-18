import os
import pandas as pd
import torch
from transformers import RobertaTokenizer, RobertaForSequenceClassification, DataCollatorWithPadding
from peft import PeftModel, PeftConfig
from torch.utils.data import DataLoader
from tqdm import tqdm
from datasets import Dataset

# 设置模型和数据路径
model_path = "results/checkpoint-1100"  # 400步的检查点路径，根据你的保存路径调整
test_data_path = r"D:\project\lalala\2025年\0416\test_unlabelled.pkl"
output_path = "submission5e-5.csv"

# 加载测试数据
print("Loading test data...")
test_dataset = Dataset.load_from_disk(test_data_path) if os.path.isdir(test_data_path) else pd.read_pickle(test_data_path)

# 检查数据格式
print(f"Test data type: {type(test_dataset)}")
if isinstance(test_dataset, pd.DataFrame):
    # 如果是DataFrame，转换为Dataset
    print(f"DataFrame columns: {test_dataset.columns}")
    test_dataset = Dataset.from_pandas(test_dataset)

# 加载tokenizer
print("Loading tokenizer...")
tokenizer = RobertaTokenizer.from_pretrained("roberta-base")

# 数据预处理函数
def preprocess(examples):
    tokenized = tokenizer(examples['text'], truncation=True, padding=True)
    return tokenized

# 对测试数据进行预处理
print("Preprocessing test data...")
tokenized_test = test_dataset.map(preprocess, batched=True, remove_columns=["text"])

# 创建数据整理器
data_collator = DataCollatorWithPadding(tokenizer=tokenizer, return_tensors="pt")

# 加载模型 - 修复的部分
print("Loading model...")
# 先加载基础模型
base_model = RobertaForSequenceClassification.from_pretrained(
    "roberta-base",
    num_labels=4,  # AG News有4个类别
    torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32
)

# 加载PEFT配置
peft_config = PeftConfig.from_pretrained(model_path)

# 加载PEFT模型
model = PeftModel.from_pretrained(base_model, model_path)

# 设置为评估模式
model.eval()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)

# 创建DataLoader
test_dataloader = DataLoader(
    tokenized_test, 
    batch_size=32, 
    collate_fn=data_collator
)

# 进行预测
print("Running inference...")
all_predictions = []

with torch.no_grad():
    for batch in tqdm(test_dataloader):
        batch = {k: v.to(device) for k, v in batch.items()}
        outputs = model(**batch)
        predictions = outputs.logits.argmax(dim=-1)
        all_predictions.extend(predictions.cpu().numpy())

# 创建提交文件
print("Creating submission file...")
submission = pd.DataFrame({
    'ID': range(len(all_predictions)),
    'Label': all_predictions
})

# 保存提交文件
submission.to_csv(output_path, index=False)
print(f"Submission saved to {output_path}")

# 打印一些样本预测结果
print("\nSample predictions:")
print(submission.head(10))

# 打印标签分布
print("\nLabel distribution:")
print(submission['Label'].value_counts())