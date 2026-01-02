# services/legacy/memory/（历史内存参考目录：已清空）

该目录用于标识 **历史的 memory 参考实现曾经所在的位置**。

当前主链路已经完成“Memory 去语义化”收敛：
- `services/memory/StandardMemAccess.{h,cc}`：主入口（实现 `api/IMemoryAccess.h`，只做 `addr+size ↔ bytes`）

> 说明：该目录刻意保持为空，以避免“legacy 复活”导致再次引入权重/BCSR 语义污染 memory 边界。

