from datasets import load_dataset


import json
import os

import os
import json
# from datasets import load_dataset # 如果您在运行此代码时需要加载数据集，请取消注释此行

def save_race_dataset_as_json(dataset, output_dir):
    """
    将RACE数据集保存为项目中使用的JSON格式。
    每个数据集分割（如 'train', 'validation', 'test'）将保存为一个独立的JSON文件。
    每个JSON文件将包含一个列表，列表中的每个元素是一个转换后的样本字典。
    
    Args:
        dataset (dict): 包含不同数据集分割（如 'train', 'validation', 'test'）的字典，
                        每个分割是一个可迭代的样本集合。
        output_dir (str): 保存JSON文件的根目录。
    """
    os.makedirs(output_dir, exist_ok=True)
    
    for split_name, split_data in dataset.items():
        # 为每个split创建一个子目录，保持与原代码的目录结构一致
        split_dir = os.path.join(output_dir, split_name)
        os.makedirs(split_dir, exist_ok=True)
        
        all_samples_for_split = [] # 用于收集当前split的所有转换后的样本
        
        for sample in split_data:
            # 转换为项目中使用的格式
            # 注意：根据ehovy/race数据集的结构，每个sample通常是一个独立的问答对。
            # 因此，这里的 'questions', 'options', 'answers' 字段会是包含单个元素的列表。
            # 例如，如果 sample["question"] 是 "What is X?", 那么 "questions" 会是 ["What is X?"]。
            # 如果 sample["options"] 是 ["A", "B", "C", "D"], 那么 "options" 会是 [["A", "B", "C", "D"]]。
            data = {
                "article": sample["article"],
                "questions": [sample["question"]],
                "options": [sample["options"]], # 这将导致 options 成为一个列表的列表，例如 [["A", "B", "C", "D"]]
                "answers": [sample["answer"]]
            }
            all_samples_for_split.append(data)
        
        # 定义输出文件名和路径，使用 .json 扩展名
        filename = f"{split_name}.json"
        filepath = os.path.join(split_dir, filename)
        
        # 将整个split的数据保存为一个JSON文件
        # ensure_ascii=False 允许非ASCII字符（如中文）直接写入
        # indent=4 使JSON输出更易读，方便阅读和调试
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(all_samples_for_split, f, ensure_ascii=False, indent=4)
        
        print(f"已将 '{split_name}' 分割保存到 '{filepath}'")

# 使用示例 (请确保您已安装 'datasets' 库: pip install datasets)
# from datasets import load_dataset

# dataset = load_dataset("ehovy/race", "high")
# save_race_dataset_as_json(dataset, "/home/ubuntu/tiansheng/26_ICLR_btk_moe/ReMoE-ts/tasks/data/RACE")

# 使用示例
dataset = load_dataset("ehovy/race", "high")
save_race_dataset_as_json(dataset, "/home/ubuntu/tiansheng/26_ICLR_btk_moe/ReMoE-ts/tasks/data/RACE")