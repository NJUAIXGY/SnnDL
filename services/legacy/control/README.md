# services/legacy/control/（历史控制面参考实现）

本目录存放 **控制面（control plane）相关的历史参考实现**，默认不参与主链路构建，仅用于：

- 对照历史行为/统计口径；
- 帮助重构时定位“口径是否漂移”；
- 临时回溯某段已被主链路吸收的实现。

> 原则：新功能禁止依赖 legacy；若确需启用，必须先做 10us→100us 回归验证，并明确回退策略。

---

## 主要内容

### `StageEventHub.{h,cc}`
- **定位**：历史的 GAS 阶段事件调度/汇报助手（BeginGather/BeginApply/BeginScatter/EndScatter）。
- **现状**：主链路已将对应能力吸收进 `control/SnnPESubComponent_impl.h` 的 `Impl`；该文件仅保留用于对照参考。

