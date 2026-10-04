# MM-OpenFGL 复现入口与数据使用手册

这份手册把论文基准代码、可确认的方法入口、四套既有五客户端划分和复现实验的防错步骤固定下来。它是**入口与数据说明**，不代表四个数据集、四条基线已经全部完成正式实验。

## 1. 官方项目与代码入口

- MM-OpenFGL 代码仓库：<https://github.com/xkLi-Allen/MM-OpenFGL>
- 作者 GitHub 主页：<https://github.com/xkLi-Allen>
- 项目文档：<https://mm-openfgl.readthedocs.io/>
- 上游 OpenFGL 仓库：<https://github.com/xkLi-Allen/OpenFGL>
- 本课题账号下的本手册：<https://github.com/schi-chexun/MM-OpenFGL-Repro-Guide>

本次核对上游 `MM-OpenFGL` 的 `main` 分支。基准运行主入口是 `src/main.py`；算法注册位于 `src/mm_openfgl/utils/basic_utils.py`；参数与数据集 YAML 由 `src/mm_openfgl/configs/config.py` 读取。

| 方法 | 上游核查结果 | 代码入口/处理 |
|---|---|---|
| FedProto | **已确认**有 CLI 选项、client/server 注册和实现目录 | `src/mm_openfgl/flcore/fedproto/client.py`、`server.py`、`fedproto_config.py`；`--fl_algorithm fedproto` |
| FedBook | 当前上游 `main` 中未定位到同名目录或注册分支 | 不以 FedLAB/SOTA、FedBookBD 或相似方法替代；需继续定位项目内别名/来源 |
| FedPG | 当前上游 `main` 中未定位到同名目录或注册分支 | 不以其他 PG/原型方法替代；需继续定位项目内别名/来源 |
| AproMFL | 当前上游 `main` 中未定位到同名目录或注册分支 | 按课题组线索继续核对项目内别名/资源；不把 FedMM、FedMosaic 等近似名称当作 AproMFL |

“没有定位到同名入口”只描述本次核对的上游分支，不等于证明方法永远不存在。每次正式实验前，应重新记录仓库 commit、注册名、实际加载类和源码文件哈希。

## 2. 四个数据集与任务

复现实验沿用服务器上已经处理好的五客户端数据，不重新下载后自行预处理、不重新划分、不触发 benchmark 的分区生成器。数据保留在课题组服务器/私有存储；本公开仓库**不包含图、特征、标签、划分张量或凭据**。

| 数据集 | 任务 | 上游数据集配置 | 已有五客户端划分目录 | 已有任务划分 |
|---|---|---|---|---|
| Toys | 节点分类 | `src/mm_openfgl/configs/dataset/toys.yaml` | `subgraph_fl_louvain_1_toys_client_5` | `node_cls/default_split/`；另有 `modal_retrieval/default_split/` |
| Grocery | 节点分类 | `src/mm_openfgl/configs/dataset/grocery.yaml` | `subgraph_fl_louvain_1_grocery_client_5` | `node_cls/default_split/` |
| Bili Music | 链路预测 | `src/mm_openfgl/configs/dataset/bili_music.yaml` | `subgraph_fl_metis_bili_music_client_5` | `link_pred/default_split/`；另有 `node_cls/default_split/` |
| DY | 链路预测 | `src/mm_openfgl/configs/dataset/dy.yaml` | `subgraph_fl_metis_dy_client_5` | `link_pred/default_split/` |

四个划分目录均应含 `data_0.pt` 至 `data_4.pt`，以及 `pre_filter.pt`、`pre_transform.pt`。任务目录保存对应任务使用的固定 split 文件。以实际服务器目录为准；表中的目录名是标识符，不是公开下载地址。

### 上游 loader 的预期路径

上游 `FGLDataset` 默认在 `--root` 下读取 `distrib/<encoder_tag>/<simulation>_<dataset>_client_<N>/data_<i>.pt`。例如，Toys 的 Louvain resolution 为 1 时会拼出 `subgraph_fl_louvain_1_toys_client_5`；Metis 链路数据按各自数据集配置和 simulation 名称解析。`encoder_tag` 从数据集 YAML 中的图像 embedding 文件名推导。

因此，运行前在服务器上先确认**现有五个 `data_i.pt` 所在的父级目录**确实与 loader 算出的路径一致，并确认任务 split 文件被该任务 evaluator 实际读取。不要只因目录名相似就继续运行。尤其当 `data_i.pt` 缺失、路径解析不同或数据校验不通过时，先停止并修复只读数据映射；不要让程序生成新划分，也不要把新划分结果与原有划分混用。

## 3. 环境与复现上游入口

以下命令按上游文档提供的接口写出；CUDA/PyTorch/PyG 版本应按服务器实际环境和官方安装说明固定。上游 README 当前建议 Python 3.10，并要求 PyTorch、PyTorch Geometric 及项目依赖。

```bash
git clone https://github.com/xkLi-Allen/MM-OpenFGL.git
cd MM-OpenFGL
git rev-parse HEAD   # 把输出的 commit 写进本次运行记录
cd src
```

上游 `src/main.py` 当前把 `args.root`、任务和算法直接赋成脚本内默认值。因此要传入本课题的数据根目录、算法和任务，**不要直接把一串 CLI 参数交给这个未修改的 `main.py` 并假定它们生效**。可使用下面的 Python 调用方式：配置模块解析 CLI 参数后，显式设置 root 再建立 trainer。`DATA_ROOT` 必须是含上游预期 `distrib/` 目录的路径，不是某个客户端文件路径。

```bash
cd MM-OpenFGL/src
DATA_ROOT=/path/to/read-only/benchmark-data \
python - --dataset_name toys --task node_cls --fl_algorithm fedproto \
  --simulation_mode subgraph_fl_louvain --louvain_resolution 1 \
  --num_clients 5 --train_val_test default_split --seed 42 <<'PY'
import os
import mm_openfgl.configs.config as config
from mm_openfgl.trainers import load_trainer

args = config.args
args.root = os.environ["DATA_ROOT"]
args.num_runs = 1
trainer = load_trainer(args)
trainer.train()
PY
```

这段示例只演示 **FedProto + Toys 节点分类** 的上游入口形式；它不是已验证过的完整正式运行命令。训练轮数、模型、学习率、batch size、FedProto 私有参数和评价方式，应从目标论文附录/课题组锁定的实验标准逐项明确传入并记录。示例中的 `seed=42` 仅用于说明，正式矩阵的种子列表由实验方案指定。

其他数据集运行前需逐项替换并确认：

| 数据集 | `--dataset_name` | `--task` | simulation 参数 |
|---|---|---|---|
| Toys | `toys` | `node_cls` | `subgraph_fl_louvain`，`--louvain_resolution 1` |
| Grocery | `grocery` | `node_cls` | `subgraph_fl_louvain`，`--louvain_resolution 1` |
| Bili Music | `bili_music` | `link_pred` | `subgraph_fl_metis` |
| DY | `dy` | `link_pred` | `subgraph_fl_metis` |

上游 YAML 配置文件存在，并不自动证明已有 server split 能被完整使用。特别是链路预测必须确认 evaluator 从目标 `link_pred/default_split` 读取固定 train/validation/test 边，消息传递图不含验证/测试正边，且所有方法使用同一组客户端和划分。若需要修改 loader 才能读取既有任务 split，应另行提交一个可审查的 adapter/补丁；不要悄悄重分区。

## 4. 每次运行前的数据校验

在提交 GPU 作业前，把以下信息写入本次 run manifest：

1. 上游仓库 URL、commit、工作区是否干净，真实算法注册名、加载类路径和对应文件 SHA-256。
2. 四个客户端文件及任务 split 文件的相对路径、大小和 SHA-256；只记录哈希与元数据，不上传数据。
3. 每个客户端的节点数、边数、特征维度、标签数量；链路任务另记录 train/validation/test 正负边数量及 split 互斥检查。
4. 数据集、任务、5 客户端、现成 simulation/split 标识、seed、模型、完整参数、运行命令、GPU、Python/PyTorch/PyG 版本。
5. 证明程序读到的是已有 `data_0.pt`–`data_4.pt` 和固定 split，而不是重新生成或回退到别的数据。预检失败即停止，不产出可比较结果。

所有模型选择只使用验证集；测试集只用于最终一次报告。节点分类至少报告 Accuracy 与 Macro-F1；链路预测报告 AUROC 与 AP。总表同时保存逐客户端值、客户端平均与 sample-weighted 汇总，并明示 bottom-20% 聚合算法。不要仅凭一条总体分数判断复现成功。

## 5. 结果与代码来源的标注规则

- 每个结果绑定到方法源码、数据与 split 哈希、参数、seed、验证选轮规则和评估脚本。
- 如果某个结果来自本地修复分支、dirty checkout 或非上游适配实现，应标为“候选/诊断”，不能称为上游 benchmark 官方结果。
- Toys FedProto 历史记录 `round 19: Accuracy 0.7094 / Macro-F1 0.6531` 是既有候选记录。它的代码来源不是已确认的干净上游 `MM-OpenFGL` checkout，因此这组数值不能单独证明上游 FedProto 复现已验收，也不能替代其他三个数据集的结果。
- 当前状态：FedProto 上游入口已核实；FedBook、FedPG、AproMFL 的上游入口及四数据集正式结果仍待逐一定位/验证。本手册没有填造缺失结果。

## 6. 官方源码核对链接

- [MM-OpenFGL README](https://github.com/xkLi-Allen/MM-OpenFGL/blob/main/README.md)
- [配置与参数解析](https://github.com/xkLi-Allen/MM-OpenFGL/blob/main/src/mm_openfgl/configs/config.py)
- [训练主入口](https://github.com/xkLi-Allen/MM-OpenFGL/blob/main/src/main.py)
- [client/server 注册器](https://github.com/xkLi-Allen/MM-OpenFGL/blob/main/src/mm_openfgl/utils/basic_utils.py)
- [FedProto 实现目录](https://github.com/xkLi-Allen/MM-OpenFGL/tree/main/src/mm_openfgl/flcore/fedproto)
- [分布式数据加载器](https://github.com/xkLi-Allen/MM-OpenFGL/blob/main/src/mm_openfgl/data/distributed_dataset_loader.py)
- [Toys 配置](https://github.com/xkLi-Allen/MM-OpenFGL/blob/main/src/mm_openfgl/configs/dataset/toys.yaml)、[Grocery 配置](https://github.com/xkLi-Allen/MM-OpenFGL/blob/main/src/mm_openfgl/configs/dataset/grocery.yaml)、[Bili Music 配置](https://github.com/xkLi-Allen/MM-OpenFGL/blob/main/src/mm_openfgl/configs/dataset/bili_music.yaml)、[DY 配置](https://github.com/xkLi-Allen/MM-OpenFGL/blob/main/src/mm_openfgl/configs/dataset/dy.yaml)

如果论文仓库链接失效，从本手册的上游链接开始，再按源码路径和 commit 核对；不要依赖临时聊天记录、匿名过期链接或他人本地目录。
