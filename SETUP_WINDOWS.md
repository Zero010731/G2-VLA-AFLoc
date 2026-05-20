# AFLoc Windows 环境部署记录

本文档记录当前机器上 AFLoc 的可运行环境。当前目标不是复现实验结果，而是先把官方模型加载、依赖导入、GPU 前向推理链路跑通。

## 1. 当前目录

仓库位置：

```powershell
C:\Users\joker\Desktop\AFLoc
```

Conda 环境：

```powershell
AFLoc
```

Python 版本：

```text
Python 3.9
```

GPU：

```text
NVIDIA GeForce RTX 4060
```

## 2. 为什么没有完全照官方 README 安装

官方 README 使用的是较旧组合：

```text
torch==1.8.0+cu111
torchvision==0.9.0+cu111
pytorch-lightning==1.1.4
```

RTX 4060 对旧 CUDA / 旧 PyTorch 兼容性不稳，所以本机采用了更适合当前显卡的组合：

```text
torch==2.5.1+cu121
torchvision==0.20.1+cu121
torchaudio==2.5.1+cu121
pytorch-lightning==1.9.5
```

另外，`omegaconf==2.0.5` 的旧包元数据会被新版 pip 拒绝，因此当前环境将 pip 固定为：

```text
pip==24.0
```

## 3. 重新创建环境的命令

```powershell
conda create -y -n AFLoc python=3.9
conda activate AFLoc

python -m pip install pip==24.0
pip install torch==2.5.1+cu121 torchvision==0.20.1+cu121 torchaudio==2.5.1+cu121 --index-url https://download.pytorch.org/whl/cu121
pip install -r requirements_windows_afloc.txt
```

## 4. 权重文件

官方 Google Drive 文件已下载到：

```text
C:\Users\joker\Desktop\AFLoc\weight
```

目前本研究优先使用胸片权重：

```text
C:\Users\joker\Desktop\AFLoc\weight\pretrained\Pretrained_CXR.ckpt
```

已下载文件包括：

```text
weight\pretrained\Pretrained_CXR.ckpt
weight\pretrained\Pretrained_Fundus.ckpt
weight\pretrained\Pretrained_Path.ckpt
weight\preprocess\MS-CXR.json
weight\preprocess\View.json
weight\preprocess\preprocess.py
weight\preprocess\resize.py
```

## 5. 已做的兼容性修补

文件：

```text
C:\Users\joker\Desktop\AFLoc\afloc\builder.py
```

原因：

旧 checkpoint 中保存的 OmegaConf 配置对象在当前 Python / OmegaConf 组合下反序列化会触发：

```text
TypeError: issubclass() arg 1 must be a class
```

处理：

在 `load_model` 和 `build_model_from_ckpt` 中加入 `_normalize_ckpt_cfg`，把旧 checkpoint 里的配置恢复成新的 `OmegaConf` 对象。修补后模型可以正常加载到 CUDA。

## 6. 环境验证

运行：

```powershell
cd C:\Users\joker\Desktop\AFLoc
conda run -n AFLoc python scripts\smoke_test_afloc.py
```

成功时会输出类似：

```json
{
  "status": "ok",
  "cuda_available": true,
  "gpu": "NVIDIA GeForce RTX 4060",
  "device": "cuda:0"
}
```

这个脚本只用于验证环境链路：导入、加载权重、处理图片、处理文本、跑一次 forward。默认图片是仓库 `assets` 里的示例图，不用于临床判断。

## 7. 单图热力图推理

运行：

```powershell
cd C:\Users\joker\Desktop\AFLoc
conda run -n AFLoc python scripts\infer_cxr_heatmap.py --image "你的单张胸片路径.png" --prompt "pleural effusion"
```

注意：`assets\results_cxr.jpg`、`assets\viz_cxr.png` 是论文展示拼图，不是单张胸片，不能作为有效定位输入。单图推理必须使用一张真实 CXR 图像，例如 MIMIC-CXR、CheXpert、NIH ChestXray14 或你自己整理出的单张胸片 PNG/JPG。

输出目录：

```text
C:\Users\joker\Desktop\AFLoc\outputs\cxr_heatmaps
```

输出文件包括：

```text
*_preprocessed.png  进入模型前的 224x224 图像空间
*_heatmap.png       AFLoc patch-text similarity heatmap
*_overlay.png       原图与热力图叠加
*_meta.json         相似度、checkpoint、prompt、张量形状等元数据
```

注意：发布的 CXR 权重使用 ClinicalBERT 文本编码器，prompt 建议先用英文疾病词或短句，例如：

```text
pleural effusion
pneumothorax
lung opacity
consolidation
atelectasis
```

## 8. 后续做正式评估还需要什么

官方 `classification.py` 和 `localization.py` 依赖具体数据集路径，仓库里存在一些 Linux 绝对路径常量。正式跑 MIMIC-CXR、MS-CXR、RSNA、CheXlocalize 等数据前，需要修改这些文件里的本地路径：

```text
afloc\constants.py
classification\constants.py
localization\constants.py
```

如果只是先做自己的研究原型，可以先绕过官方评估脚本，直接使用：

```text
afloc.builder.load_model
model.process_img
model.process_text
model.forward
```

来写自定义图像-文本或图像-病例图对齐实验脚本。

## 9. 下载 MS-CXR 标注文件

MS-CXR 是 PhysioNet credentialed-access 数据集，需要你自己的 PhysioNet 账号已完成 credentialing 并签署 DUA。当前机器没有配置 PhysioNet 登录凭据，直接下载会返回 `403 Forbidden`。

如果账号已经有权限，可以运行：

```powershell
cd C:\Users\joker\Desktop\AFLoc
powershell -ExecutionPolicy Bypass -File scripts\download_ms_cxr.ps1
```

下载目标：

```text
C:\Users\joker\Desktop\AFLoc\data\ms-cxr\1.1.0
```

会下载：

```text
MS_CXR_Local_Alignment_v1.1.0.json
MS_CXR_Local_Alignment_v1.1.0.csv
convert_coco_json_to_csv.py
```

注意：MS-CXR 只提供短语-框标注，不包含胸片图像本体。真正的图像需要从 MIMIC-CXR-JPG 另行下载，然后把 `localization\constants.py` 里的 `MIMIC_IMG_DIR` 指向本地 MIMIC-CXR-JPG 图像目录。

当前项目内已经支持用环境变量覆盖路径，不必每次手改代码：

```powershell
$env:MS_CXR_JSON = "C:\Users\joker\Desktop\AFLoc\data\ms-cxr\1.1.0\MS_CXR_Local_Alignment_v1.1.0.json"
$env:MIMIC_CXR_JPG_ROOT = "你的MIMIC-CXR-JPG图像根目录"
```

MS-CXR 标注里的图像路径长这样：

```text
files/p10/p10233088/s54276838/675d792f-a3521e48-5eec8573-1e81d644-e60c34f8.jpg
```

AFLoc 的 MS-CXR loader 会自动去掉开头的 `files/`，所以如果你的真实图片是：

```text
/mnt/mimic-cxr/jpg/files/p10/...
```

那么应设置：

```powershell
$env:MIMIC_CXR_JPG_ROOT = "/mnt/mimic-cxr/jpg/files"
```

如果你的真实图片是：

```text
/mnt/mimic-cxr/jpg/p10/...
```

那么应设置：

```powershell
$env:MIMIC_CXR_JPG_ROOT = "/mnt/mimic-cxr/jpg"
```

可以先用路径探测脚本确认哪一层正确：

```powershell
conda run -n AFLoc python scripts\check_ms_cxr_paths.py --mimic-root "你的MIMIC-CXR-JPG候选根目录"
```

服务器上可以使用模板脚本：

```bash
bash scripts/run_ms_cxr_server.sh
```

如果服务器真实图像根目录不是默认值，可以临时覆盖：

```bash
MS_CXR_JSON=/path/to/MS_CXR_Local_Alignment_v1.1.0.json \
MIMIC_CXR_JPG_ROOT=/mnt/mimic-cxr/jpg/files \
GPU=0 \
bash scripts/run_ms_cxr_server.sh
```
