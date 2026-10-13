# Changelog

## v1.0.0 — 2026-10-13

首个公开的 CPU 研究原型版本；版本号表示工程里程碑，不表示产线验收通过。

- 轻量 patch memory 检测、热力图、候选缺陷框。
- 独立概率拟合、标签条件 conformal、独立漏检风险验收。
- FastAPI、中文 Streamlit、SQLite 复核及标签修改历史。
- 随机 / 不确定性 / coreset 选样，候选训练、新独立数据重新校准。
- 五种公开数据格式导入；KolektorSDD2 完整官方测试集实测。
- CPU ONNX 导出与数值一致性验证；Windows 启停脚本和 Docker 配置。
- 18 项本地测试通过；新增 Windows / Linux GitHub Actions CPU 检查。

KolektorSDD2 主检测 AUROC 为 0.928229。独立路由未通过 1% 漏检风险验收，自动放行关闭。深度 PatchCore/EfficientAD 仅提供未验证的训练入口，尚未接入默认服务。

完整范围见 [v1 发布说明](docs/releases/v1.0.0.md)，下一阶段见 [v2 加强版规划](docs/v2加强版规划.md)。
