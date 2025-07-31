#!/usr/bin/env python
"""Download the AI2 ARC dataset using the 🤗 datasets library
and save it to a local directory for downstream evaluation.

用法：
    python tasks/data/get_arc.py \
        --subset arc-e \
        --split validation \
        --output_dir tasks/data/arc

参数说明：
    --subset:      ARC子集，可选 [arc-e, arc-c]
    --split:       要下载的数据划分，可选 [train, validation, test]
    --output_dir:  保存数据集的本地目录。

依赖：pip install datasets tqdm
"""
import argparse
import json
import os
from pathlib import Path

from datasets import load_dataset
from datasets import disable_caching


def parse_args():
    parser = argparse.ArgumentParser(description="Download AI2 ARC dataset")
    parser.add_argument(
        "--subset",
        default='ARC-Challenge',
        choices=['ARC-Challenge', 'ARC-Easy'],
        help="ARC子集 (arc-e/arc-c)。",
    )
    parser.add_argument(
        "--split",
        default="validation",
        choices=["train", "validation", "test"],
        help="要下载的数据集划分 (train/validation/test)。",
    )
    parser.add_argument(
        "--output_dir",
        default="./arc",
        help="保存数据集的本地目录。",
    )
    return parser.parse_args()


def save_arc_dataset_as_json(dataset, output_dir, subset_name):
    """
    将ARC数据集保存为项目中使用的JSON格式。
    每个数据集分割将保存为一个独立的JSON文件。
    
    Args:
        dataset: 数据集对象
        output_dir: 保存JSON文件的根目录
        subset_name: 子集名称 (arc-e 或 arc-c)
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # 为每个split创建一个子目录
    split_dir = os.path.join(output_dir, subset_name)
    os.makedirs(split_dir, exist_ok=True)
    
    all_samples_for_split = []
    
    for sample in dataset:
        # 转换为项目中使用的格式
        data = {
            "question": sample["question"],
            "choices": sample["choices"]["text"],  # ARC的选项格式
            "answerKey": sample["answerKey"],      # 正确答案的key
            "id": sample.get("id", ""),
            "category": sample.get("category", "")
        }
        all_samples_for_split.append(data)
    
    # 定义输出文件名和路径
    filename = f"{subset_name}.json"
    filepath = os.path.join(split_dir, filename)
    
    # 保存为JSON文件
    with open(filepath, 'w', encoding='utf-8') as f:
        json.dump(all_samples_for_split, f, ensure_ascii=False, indent=4)
    
    print(f"已将 '{subset_name}' 分割保存到 '{filepath}'")
    print(f"总共 {len(all_samples_for_split)} 个样本")


def main():
    args = parse_args()
    # 关闭 datasets 的缓存以避免混淆
    disable_caching()

    print(f"开始下载 AI2 ARC ({args.subset}, {args.split}) …")
    
    # 加载ARC数据集
    dataset = load_dataset("ai2_arc", args.subset, split=args.split)

    output_dir = Path(args.output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"保存到 {output_dir} …")
    save_arc_dataset_as_json(dataset, output_dir, args.subset)
    print("完成！")


if __name__ == "__main__":
    main()
