# SnnDL Submission Package

本目录为提交到 GitHub 的最小自包含包，包含：

- `SnnDL/`：SST 元素库 SnnDL 的源码（仅源码，已剔除构建产物）。
- `SnnDL_Basic/`：规范的 4x4 分层分类示例与脚本（后续开发的基线模板）。
- `experimental_features/`：实验性/研究性功能与示例（映射框架、学习测试、网络校验等）。

## 目录结构

```
github_submission/
├── SnnDL/                       # SST 元素：SnnDL 源码（放入 sst-elements 后构建）
├── SnnDL_Basic/                 # 规范 4x4 分类示例（脚本、权重、脉冲数据、分析）
└── experimental_features/
    ├── neuron_mapping_framework # 神经元到 PE 映射框架（独立 C++ 项目）
    ├── snnDL_learning_tests     # 学习特性实验脚本（试验性）
    ├── snnDL_network_validator  # 网络配置校验工具（试验性）
    └── snnDL_neuron_dynamics_tests # 神经元动力学实验脚本（试验性）
```

## 环境与依赖

- 已安装或可构建的 SST 框架（sst-core 与 sst-elements）。
- 推荐安装前缀：`/home/<user>/SST/sst_install`（如下示例可替换为你的路径）。
- 运行 Python 脚本需要 Python 3.x。

## 构建 SnnDL 元素

方式 A：在已有的 sst-elements 工作区内增量构建（推荐）

1) 将本目录下的 `SnnDL/` 拷贝至你的 sst-elements：

```
cp -a github_submission/SnnDL <你的>/sst-elements/src/sst/elements/SnnDL
```

2) 安装 sst-core 与 sst-elements（如已安装可跳过）：

```
cd <你的>/sst-core
./configure --prefix=/home/<user>/SST/sst_install
make -j4 && make install

cd <你的>/sst-elements
./configure --prefix=/home/<user>/SST/sst_install --with-sst-core=/home/<user>/SST/sst_install
make -j4 && make install
```

3) 增量构建 SnnDL 元素：

```
cd <你的>/sst-elements/src/sst/elements/SnnDL
make -j4 && make install
```

方式 B：全量构建（当需要重新配置/安装整个 SST 环境时）

参考方式 A 的完整 configure + make + install 流程。

提示：运行时确保环境变量可找到安装位置：

```
export PATH=/home/<user>/SST/sst_install/bin:$PATH
export LD_LIBRARY_PATH=/home/<user>/SST/sst_install/lib:/home/<user>/SST/sst_install/lib/sst-elements-library:$LD_LIBRARY_PATH
```

## 运行 4x4 分类示例（规范模板）

```
cd github_submission/SnnDL_Basic/scripts
sst test_classification_4x4.py

# 分析输出（CSV 写在当前目录）
python analyze_classification_results.py
```

说明：示例脚本使用就地输出 `./complex_classification_stats.csv`，不依赖外部 `sst_output_data/` 目录。

## 实验性功能（experimental_features）

1) 映射框架（neuron_mapping_framework）编译校验：

```
cd github_submission/experimental_features/neuron_mapping_framework
make test-compile
```

2) 学习测试（snnDL_learning_tests）：

- 示例脚本 `scripts/test_learning_min.py` 使用绝对路径指向本仓 `experimental_features/snnDL_learning_tests` 下的数据/误差文件。
- 若你在不同路径运行，请相应调整脚本中的路径参数或在根目录下运行以保持路径一致。

## 常见问题

- 若 `sst` 运行时报 MPI/PMIx 相关错误，通常是环境限制导致。可在支持 OpenMPI/PMIx 的环境下运行，或在简单用例中设置单分区（`sst.single`）后再试（并非所有脚本都适用）。
- 如需更换安装前缀，请同步调整环境变量 `PATH` 与 `LD_LIBRARY_PATH`。

## 许可与贡献

- 源码以研究/实验为主，欢迎在独立分支上提交 PR 讨论改进。

