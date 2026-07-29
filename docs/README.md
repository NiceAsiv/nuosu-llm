# Documentation map / 文档地图

## 推荐阅读顺序

1. [`01-scope-and-architecture.md`](01-scope-and-architecture.md)：语言范围、仓库边界与模型路线；
2. [`../scripts/model/README.md`](../scripts/model/README.md)：安全获取和验证基础模型；
3. [`03-training-guide.md`](03-training-guide.md)：CPT、SFT、多卡、恢复和训练门禁；
4. [`04-evaluation.md`](04-evaluation.md)：自动指标与母语者盲评；
5. [`05-deployment.md`](05-deployment.md)：合并 adapter、服务化和性能测试；
6. [`06-training-server.md`](06-training-server.md)：服务器目录和硬件检查；
7. [`09-publishing.md`](09-publishing.md)：GitHub、Hugging Face 与版本关联。

## 按角色

| 角色 | 必读 |
|---|---|
| 数据贡献者 | `nuosu-corpus` README、范围与架构 |
| 训练执行者 | 模型恢复、训练指南、配置目录 |
| 评测者 | 评测方案、`evaluation/README.md` |
| 发布维护者 | 部署指南、发布指南、model card 模板 |

实验报告位于 [`../evaluation/reports/`](../evaluation/reports/)，不与稳定使用说明混写。
