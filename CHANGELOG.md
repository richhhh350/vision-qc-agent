# Changelog

## v2.0.0 — 2026-10-14（本地研究版）

- Anomalib 2.2.0 PatchCore 与 CPU 预算 EfficientAD 接入统一原图坐标推理契约。
- 等比例 256/384 与重叠切片；原尺寸 Pixel AUROC / AUPRO 近似指标及小区域命中统计。
- dev / acceptance 分离，解码像素去重、独立工件计数约束、完整策略证书检查与新证据重校准。
- 小型 U-Net、掩膜上传与修订绑定；正常反馈更新一级、像素缺陷反馈训练二级，候选训练后台运行并支持取消。
- KSDD2 真实池上 4 策略 × 3 种子 × 3 预算实验；公开选样概率、标签成本、逐轮结果及未达标记录。
- ONNX 原尺寸热力图/分数/阈值附近路由核验，SQLite 模型注册、提升与回滚，v1 接口回归。
- Windows 依赖版本快照、深度契约 CI、完整 HTTP 压测脚本和 v2 Docker 配置。

研究目标与验证限制见 [v2 验证报告](docs/v2验证报告.md)。本版本不能声称已满足 1% 漏检风险、降低复核率或主动学习节省标注成本；二级模型仅提供人工参考。

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
