# VLM 分类项目

本项目专注于基于视觉语言模型（VLM）的分类任务，特别是针对与甲状腺相关眼病（TAO）相关的医学影像分析。它利用Swin Transformer和基础模型等深度学习模型对医学扫描影像进行分类。

---

## 详细文件结构

```
E:\zizhu\MLLM\VLM_classification\
├── main_Swin_TAO_CLS.py
├── trainer_TAO_CLS.py
├── README.md
├── README_zh.md
├── dataset\
│   └── mm_tao_cls\
│       ├── data_split_with_csv.py
│       ├── json_combine.py
│       ├── mm_tao_cls_4_label_test.json
│       ├── mm_tao_cls_4_label.csv
│       ├── mm_tao_cls_4_label.json
│       ├── mm_tao_cls_label.csv
│       ├── mm_tao_list_test.json
│       ├── mm_tao_list.json
│       ├── mm_tao_test_list_4_label.json
│       ├── mm_tao_train_list_4_label.json
│       ├── mm_tao_val_list_4_label.json
│       ├── test.csv
│       ├── train.csv
│       ├── val.csv
│       ├── test_data\images
│       ├── train_data\images
│       ├── train_data\masks
│       └── val_data\images
│       └── val_data\masks
├── model\
│   └── CLS\
│       ├── 3D_Classification_dual_Att.py
│       ├── Classification_model.py
│       ├── mm_classification_Foundation_model_plus.py
│       ├── mm_classification_Foundation_model.py
│       ├── mm_classification_model.py
│       ├── mm_classification_SwinUnter.py
│       ├── resnet.py
│       └── transformer_decoder.py
├── optimizers\
│   ├── lr_scheduler.py
│   └── segment_anything\
│       ├── build_sam.py
│       ├── predictor.py
│       └── ...
├── pretrained_models\
│   └── Foundation_model.pth
├── runs\
│   └── __init__.py
├── Text-emmbedding-gen\
│   └── TAO_bert_txt_encoding.pth
└── utils\
    ├── data_utils_mm.py
    ├── data_utils.py
    ├── Focal_Loss.py
    ├── loss.py
    ├── MM_CLS_Brain_data_utils.py
    ├── MM_CLS_Liver_data_utils.py
    ├── MM_CLS_TAO_data_utils_v1.py
    ├── MM_CLS_TAO_data_utils.py
    ├── pretrain_dataset.py
    ├── Semi_MM_CLS_data_utils.py
    ├── Semi_MM_data_utils.py
    ├── test.py
    └── utils.py
```

---

## `dataset` 目录详解

本目录存放项目所需的数据。理想的数据组织结构如下：

#### 1. 元数据文件 (CSV and JSON)

这些文件位于 `dataset/mm_tao_cls/` 目录下，用于定义数据集的结构、标签和划分。

- **`mm_tao_cls_4_label.csv`**: 这是最核心的原始数据表。它应包含但不限于以下列：
  - `image`: 图像文件的相对路径 (例如, `train_data/images/eye_100_t1_sequence_29_Jul_2021.nii`)
  - `label`: 图像对应的分类标签 (例如, 0, 1, 2, 3)
  - `patient_id`: 患者ID，用于标识唯一患者，方便进行按患者划分数据集。

- **`train.csv`, `val.csv`, `test.csv`**: 这些是由 `data_split_with_csv.py` 脚本从主CSV文件分割而成，用于训练、验证和测试。它们的结构与主CSV文件相同。

- **`*.json` 文件**: 这些是另一种定义数据集划分的方式，常见于许多深度学习框架。通常，一个JSON文件会包含一个列表，列表中的每个对象代表一个数据样本，格式可能如下：
  ```json
  [
      {
          "image": "path/to/image1.nii",
          "label": 1
      },
      {
          "image": "path/to/image2.nii",
          "label": 0
      }
  ]
  ```

#### 2. 图像和掩码数据

实际的图像和掩码文件存放在 `train_data`, `val_data`, `test_data` 子目录中。

- **文件格式**:
  - **图像/掩码**: `.nii` (NIfTI 格式)。这是一种在医学影像（如MRI, CT）中广泛使用的标准格式，可以存储3D或4D数据及元信息。

- **目录结构**:
  ```
  dataset/mm_tao_cls/
  ├── train_data/
  │   ├── images/
  │   │   └── eye_100_t1_sequence_29_Jul_2021.nii
  │   └── masks/
  │       └── eye_100_t1_PL_29_Jul_2021.nii
  ├── val_data/
  │   └── ...
  └── test_data/
      └── ...
  ```

- **文件命名规范**:
  - **图像**: `eye_{患者ID}_{扫描类型}_sequence_{扫描日期}.nii`
    - 例如: `eye_122_t1c_sequence_24_Nov_2022.nii`
  - **掩码**: `eye_{患者ID}_{扫描类型}_PL_{扫描日期}.nii`
    - 例如: `eye_100_t1c_PL_25_Aug_2022.nii`
  - 清晰的命名规范确保了每个图像都能和其对应的掩码（如果存在）以及元数据表中的记录相关联。

---

## 核心模块说明

#### `main_Swin_TAO_CLS.py`
用于启动基于Swin Transformer的TAO分类模型的训练和评估的主脚本。

#### `trainer_TAO_CLS.py`
该文件包含核心的训练逻辑，如训练循环、验证、损失计算和优化。

#### `model/CLS/`
包含所有神经网络模型的定义，如 `mm_classification_Foundation_model.py` 和 `mm_classification_SwinUnter.py`。

#### `utils/`
包含各类工具脚本，其中 `MM_CLS_TAO_data_utils.py` 等文件是核心的数据加载和预处理管道。