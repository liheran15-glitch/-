#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
GRAM 快速训练脚本 (Python版本)
"""

import os
import sys
import subprocess
from pathlib import Path

def print_header():
    print("=" * 50)
    print("GRAM 快速训练脚本")
    print("=" * 50)
    print()

def create_output_dirs():
    """创建输出目录"""
    output_dir = "./output/gram/pretrain_gram"
    dirs = [
        output_dir,
        f"{output_dir}/downstream",
        f"{output_dir}/downstream/pretrain"
    ]
    
    for dir_path in dirs:
        os.makedirs(dir_path, exist_ok=True)
        print(f"创建目录: {dir_path}")

def run_training():
    """运行训练"""
    print()
    print("=" * 50)
    print("开始训练 GRAM 模型")
    print("=" * 50)
    print()
    
    print("训练配置:")
    print("- 学习率: 2e-5")
    print("- GPU 数量: 4")
    print("- 配置文件: ./config/gram/finetune_cfg/pretrain-gram.json")
    print("- 输出目录: ./output/gram/pretrain_gram")
    print()
    
    # 构建训练命令
    cmd = [
        "python", "-m", "torch.distributed.launch",
        "--nnodes", "1",
        "--node_rank", "0",
        "--nproc_per_node", "4",
        "--master_port", "9834",
        "./run.py",
        "--learning_rate", "2e-5",
        "--checkpointing", "true",
        "--first_eval", "true",
        "--save_best", "true",
        "--config", "./config/gram/finetune_cfg/pretrain-gram.json",
        "--output_dir", "./output/gram/pretrain_gram"
    ]
    
    print("执行命令:")
    print(" ".join(cmd))
    print()
    
    try:
        result = subprocess.run(cmd, check=True)
        print()
        print("=" * 50)
        print("训练成功完成!")
        print("=" * 50)
        print("模型已保存到: ./output/gram/pretrain_gram")
        return True
    except subprocess.CalledProcessError as e:
        print()
        print("=" * 50)
        print("训练过程中出现错误")
        print("=" * 50)
        print(f"错误代码: {e.returncode}")
        return False
    except KeyboardInterrupt:
        print()
        print("=" * 50)
        print("训练被用户中断")
        print("=" * 50)
        return False

def main():
    print_header()
    
    # 切换到脚本所在目录
    script_dir = os.path.dirname(os.path.abspath(__file__))
    os.chdir(script_dir)
    print(f"当前目录: {os.getcwd()}")
    print()
    
    # 创建输出目录
    create_output_dirs()
    
    # 运行训练
    success = run_training()
    
    print()
    print("按 Enter 键退出...")
    input()

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"\n\n发生错误: {e}")
        import traceback
        traceback.print_exc()
        print("\n按 Enter 键退出...")
        input()
        sys.exit(1)
