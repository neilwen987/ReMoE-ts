
from datasets import load_dataset
import json

# 下载LAMBADA数据集
dataset = load_dataset("cimec/lambada", "plain_text")

# 获取测试集
test_data = dataset["test"]

# 转换为JSONL格式，每行一个JSON对象
output_file = "/home/ubuntu/tiansheng/26_ICLR_btk_moe/ReMoE-ts/tasks/data/lambada_test.jsonl"

with open(output_file, 'w', encoding='utf-8') as f:
    for example in test_data:
        # 将每个样本转换为所需的JSON格式
        json_obj = {"text": example["text"]}
        f.write(json.dumps(json_obj, ensure_ascii=False) + "\n")

print(f"LAMBADA测试集已保存到: {output_file}")
print(f"总共 {len(test_data)} 个样本")
