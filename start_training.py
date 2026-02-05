#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
GRAM 模型训练启动脚本 (Python版本)
"""

import os
import sys
import subprocess
import platform
import time
from pathlib import Path

def print_header():
    print("=" * 50)
    print("GRAM 模型训练启动脚本")
    print("作者: Trae AI Assistant")
    print(f"日期: {time.strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 50)
    print()

def check_python_environment():
    print("=" * 50)
    print("检查 Python 环境")
    print("=" * 50)
    
    # 检查 Python 版本
    print(f"Python 版本: {sys.version}")
    
    # 检查 PyTorch
    try:
        import torch
        print(f"PyTorch 版本: {torch.__version__}")
        print(f"CUDA 可用: {torch.cuda.is_available()}")
        if torch.cuda.is_available():
            print(f"CUDA 版本: {torch.version.cuda}")
            print(f"GPU 数量: {torch.cuda.device_count()}")
            for i in range(torch.cuda.device_count()):
                print(f"  GPU {i}: {torch.cuda.get_device_name(i)}")
    except ImportError:
        print("错误: 未找到 PyTorch!")
        print("请运行 preinstall.sh 安装依赖")
        return False
    
    return True

def check_pretrained_weights():
    print()
    print("=" * 50)
    print("检查预训练权重")
    print("=" * 50)
    
    required_weights = [
        "./pretrained_weights/clip/EVA01_CLIP_g_14_psz14_s11B.pt",
        "./pretrained_weights/beats/beats-base-itr3.pt",
        "./pretrained_weights/bert/bert-base-uncased/pytorch_model.bin"
    ]
    
    all_weights_exist = True
    for weight in required_weights:
        if os.path.exists(weight):
            print(f"[OK] {weight}")
        else:
            print(f"[缺失] {weight}")
            all_weights_exist = False
    
    if not all_weights_exist:
        print()
        print("错误: 缺少预训练权重!")
        print("请按照 README.md 中的说明下载所有必要的预训练权重")
        return False
    
    # 检查 GRAM 预训练模型
    if os.path.exists("./model/pretrained_models/model_step_459-001.pt"):
        print("[OK] GRAM 预训练模型已存在")
    else:
        print("[提示] GRAM 预训练模型不存在，将从头开始训练")
    
    return True

def check_system_environment():
    print()
    print("=" * 50)
    print("系统环境检查")
    print("=" * 50)
    
    print(f"操作系统: {platform.system()} {platform.release()}")
    print(f"处理器: {platform.processor()}")
    print(f"计算机名: {platform.node()}")
    print(f"Python 版本: {sys.version.split()[0]}")
    
    # 检查磁盘空间
    try:
        import shutil
        disk_usage = shutil.disk_usage(".")
        free_gb = disk_usage.free / (1024**3)
        total_gb = disk_usage.total / (1024**3)
        print(f"磁盘空间: 可用 {free_gb:.2f} GB / 总计 {total_gb:.2f} GB")
    except Exception as e:
        print(f"无法获取磁盘空间信息: {e}")
    
    print()
    print("=" * 50)
    print("依赖检查")
    print("=" * 50)
    
    if os.path.exists("./requirements.txt"):
        print("[OK] requirements.txt 存在")
        with open("./requirements.txt", 'r', encoding='utf-8') as f:
            lines = [line.strip() for line in f if line.strip() and not line.startswith('#')]
            print(f"依赖项数量: {len(lines)}")
    else:
        print("[缺失] requirements.txt")
    
    print()
    print("=" * 50)
    print("配置文件检查")
    print("=" * 50)
    
    if os.path.exists("./config/gram/finetune_cfg/pretrain-gram.json"):
        print("[OK] 预训练配置文件存在")
    else:
        print("[缺失] 预训练配置文件不存在")

def run_pretrain():
    print()
    print("=" * 50)
    print("开始预训练")
    print("=" * 50)
    print("正在执行预训练命令...")
    print()
    
    # 创建输出目录
    output_dir = "./output/gram/pretrain_gram"
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(f"{output_dir}/downstream", exist_ok=True)
    os.makedirs(f"{output_dir}/downstream/pretrain", exist_ok=True)
    
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
        "--output_dir", output_dir
    ]
    
    print("执行命令:")
    print(" ".join(cmd))
    print()
    
    try:
        result = subprocess.run(cmd, check=True)
        print()
        print("=" * 50)
        print("预训练完成")
        print("=" * 50)
        return True
    except subprocess.CalledProcessError as e:
        print()
        print("=" * 50)
        print("预训练失败")
        print("=" * 50)
        print(f"错误代码: {e.returncode}")
        return False

def run_downstream_task(task_name, config_file, learning_rate="2e-5", port_offset=1):
    print()
    print(f"执行 {task_name} 检索任务微调...")
    print()
    
    output_dir = "./output/gram/pretrain_gram"
    master_port = 9834 + port_offset
    
    cmd = [
        "python", "-m", "torch.distributed.launch",
        "--nnodes", "1",
        "--node_rank", "0",
        "--nproc_per_node", "4",
        "--master_port", str(master_port),
        "./run.py",
        "--learning_rate", learning_rate,
        "--checkpointing", "true",
        "--config", config_file,
        "--save_best", "true",
        "--output_dir", f"{output_dir}/downstream/{task_name}"
    ]
    
    print("执行命令:")
    print(" ".join(cmd))
    print()
    
    try:
        result = subprocess.run(cmd, check=True)
        print()
        print(f"{task_name} 微调完成")
        return True
    except subprocess.CalledProcessError as e:
        print()
        print(f"{task_name} 微调失败")
        print(f"错误代码: {e.returncode}")
        return False

def show_menu():
    print()
    print("=" * 50)
    print("训练选项")
    print("=" * 50)
    print("1. 开始预训练 (VAST27M 数据集)")
    print("2. 运行完整训练脚本 (包含下游任务选项)")
    print("3. 检查系统环境")
    print("4. 退出")
    print()

def show_downstream_menu():
    print()
    print("=" * 50)
    print("下游任务选项")
    print("=" * 50)
    print("1. VATEX 检索任务")
    print("2. YouCook 检索任务")
    print("3. DiDeMo 检索任务")
    print("4. ActivityNet 检索任务")
    print("5. 返回主菜单")
    print()

def main():
    print_header()
    
    # 切换到脚本所在目录
    script_dir = os.path.dirname(os.path.abspath(__file__))
    os.chdir(script_dir)
    print(f"当前目录: {os.getcwd()}")
    print()
    
    # 检查环境
    if not check_python_environment():
        input("按 Enter 键退出...")
        return
    
    if not check_pretrained_weights():
        input("按 Enter 键退出...")
        return
    
    while True:
        show_menu()
        choice = input("请选择一个选项 (1-4): ").strip()
        
        if choice == "1":
            run_pretrain()
        elif choice == "2":
            print()
            print("运行完整训练脚本...")
            
            # 先运行预训练
            if run_pretrain():
                print()
                print("预训练完成，现在可以选择下游任务...")
                
                while True:
                    show_downstream_menu()
                    downstream_choice = input("请选择下游任务 (1-5): ").strip()
                    
                    if downstream_choice == "1":
                        run_downstream_task("retrieval-vatex", 
                                          "./config/gram/finetune_cfg/retrieval-vatex.json",
                                          "2e-5", 1)
                    elif downstream_choice == "2":
                        run_downstream_task("retrieval-youcook", 
                                          "./config/gram/finetune_cfg/retrieval-youcook.json",
                                          "3e-5", 2)
                    elif downstream_choice == "3":
                        run_downstream_task("retrieval-didemo", 
                                          "./config/gram/finetune_cfg/retrieval-didemo.json",
                                          "2e-5", 3)
                    elif downstream_choice == "4":
                        run_downstream_task("retrieval-activitynet", 
                                          "./config/gram/finetune_cfg/retrieval-activitynet.json",
                                          "2e-5", 4)
                    elif downstream_choice == "5":
                        break
                    else:
                        print("无效选项，请重新选择")
        elif choice == "3":
            check_system_environment()
            input("\n按 Enter 键返回主菜单...")
        elif choice == "4":
            print()
            print("=" * 50)
            print("退出脚本")
            print("=" * 50)
            print("训练脚本已退出")
            break
        else:
            print("无效选项，请输入 1-4 之间的数字")
    
    print()
    print("按 Enter 键退出...")
    input()

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n程序被用户中断")
        sys.exit(0)
    except Exception as e:
        print(f"\n\n发生错误: {e}")
        import traceback
        traceback.print_exc()
        input("\n按 Enter 键退出...")
        sys.exit(1)
