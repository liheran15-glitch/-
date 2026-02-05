# GRAM 训练问题解决方案

## 当前问题

运行训练脚本时出现错误：
```
FileNotFoundError: [Errno 2] No such file or directory: './output/gram/pretrain_gram\\log\\hps.json'
```

## 问题原因

1. **配置文件中的数据路径问题**：
   - 配置文件 `config/gram/finetune_cfg/pretrain-gram.json` 中的数据路径是Linux路径
   - 例如：`/leonardo_work/IscrC_GenOpt/datasets/vast27m/videos/`
   - 这些路径在Windows上不存在

2. **缺少训练数据集**：
   - 需要下载相应的数据集（VAST27M、MSRVTT等）
   - 数据集需要包含视频、音频和标注文件

## 解决方案

### 方案 1：使用预训练模型进行推理（推荐）

如果你只是想测试模型，可以使用已有的预训练模型进行推理：

```python
from utils.utils_for_fast_inference import get_args, VisionMapper, AudioMapper, build_batch
from utils.build_model import build_model
from utils.volume import volume_computation3
import warnings
import os
warnings.filterwarnings("ignore")

os.environ['LOCAL_RANK'] = '0'

# 使用已有的预训练模型
pretrain_dir = './model/pretrained_models'

args = get_args(pretrain_dir)
model,_,_ = build_model(args)
model.to('cuda')

visionMapper = VisionMapper(args.data_cfg.train[0],args)
audioMapper = AudioMapper(args.data_cfg.train[0],args)

tasks = 'ret%tva'

text = ["A dog is barking","A dog is howling", "A red cat is meowing", "A black cat is meowing"]
video = ["./assets/videos/video1.mp4","./assets/videos/video2.mp4","assets/videos/video3.mp4","./assets/videos/video4.mp4"]
audio = ["./assets/audios/audio1.mp3","./assets/audios/audio2.mp3","./assets/audios/audio3.mp3","./assets/audios/audio4.mp3"]

batch = build_batch(args,text,video,audio)

evaluation_dict = model(batch, tasks, compute_loss=False)

feat_t = evaluation_dict['feat_t']
feat_v = evaluation_dict['feat_v']
feat_a = evaluation_dict['feat_a']

volume = volume_computation3(feat_t,feat_v,feat_a)

print("Volume: ", volume.detach().cpu())
```

### 方案 2：下载训练数据集并修改配置

如果你想进行训练，需要：

1. **下载训练数据集**：
   - VAST27M 数据集：https://github.com/TXH-mercury/VAST
   - MSRVTT 数据集：需要从官方渠道下载
   - 其他数据集：根据需要下载

2. **修改配置文件**：
   - 编辑 `config/gram/finetune_cfg/pretrain-gram.json`
   - 将Linux路径改为Windows路径
   - 例如：
     ```json
     "txt": "C:\\path\\to\\your\\data\\annotations150k.json",
     "vision": "C:\\path\\to\\your\\data\\videos\\",
     "audio": "C:\\path\\to\\your\\data\\audios"
     ```

3. **运行训练**：
   ```bash
   python quick_train.py
   ```

### 方案 3：使用示例数据进行测试

如果你想快速测试训练流程，可以使用项目中的示例数据：

1. **创建测试配置文件**：
   - 复制 `config/gram/finetune_cfg/pretrain-gram.json`
   - 修改数据路径指向 `assets/` 目录

2. **修改配置示例**：
   ```json
   {
     "run_cfg": {"default":"./config/gram/default_run_cfg.json"},
     "model_cfg": {"default":"./config/gram/default_model_cfg.json"},
     "data_cfg": {
       "train": [{
         "type":"annoindexed",
         "training":true,
         "name": "test_data",
         "txt": "./assets/videos",  # 使用示例数据
         "vision": "./assets/videos",
         "audio": "./assets/audios",
         "vision_transforms":"crop_flip",
         "vision_format": "video_rawvideo",
         "vision_sample_num": 2,
         "audio_sample_num": 1,
         "task": "ret%tv%ta",
         "epoch": 1,
         "n_workers": 2,
         "batch_size": 4
       }],
       "val": []
     }
   }
   ```

## 推荐步骤

1. **首先尝试方案1**：使用预训练模型进行推理
2. **如果需要训练**：下载完整的数据集并使用方案2
3. **快速测试**：使用方案3测试训练流程

## 注意事项

- 训练需要大量磁盘空间和计算资源
- 数据集下载可能需要较长时间
- 确保Python环境和依赖都已正确安装
- 检查GPU是否可用（`torch.cuda.is_available()`）

## 获取帮助

如果遇到问题，请检查：
1. Python环境是否正确
2. 预训练权重是否已下载
3. 数据集路径是否正确
4. 配置文件格式是否正确
