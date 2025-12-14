# components/mpi/（MPI 扩展组件）

本目录存放 **SnnDL 的 MPI 扩展实现**。其目标是为多进程/多节点环境提供可选的接口与组件，但不应影响默认（非 MPI）路径的可构建与可运行性。

## 主要内容

- `MPITypes.{h,cc}`：MPI 类型/封装（用于与 SST/MPI 运行环境协同）。
- `MPIMultiCorePE.{h,cc}`：MPI 版本的 PE 组件或扩展入口（视当前实现而定）。

## 依赖与约束

- 本目录代码可能依赖 MPI 头与链接选项；默认构建应尽量保持可选（不强制要求 MPI 环境）。
- 若启用 MPI，建议在 configure/Makefile 层面显式开关，并保持与非 MPI 路径行为一致。

