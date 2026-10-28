"""Start with: streamlit run dashboard.py"""
import os

import httpx
import pandas as pd
import streamlit as st

st.set_page_config(page_title="视觉质检 Agent", page_icon="◉", layout="wide")
st.markdown("""<style>
.stApp {background:#f6f8fb;} h1,h2,h3 {letter-spacing:-.03em;}
[data-testid="stMetric"] {background:white;padding:18px;border:1px solid #e3e8ef;border-radius:12px;}
[data-testid="stSidebar"] {background:#eaf0f6;}
</style>""", unsafe_allow_html=True)
st.title("视觉质检 Agent")
st.caption("异常检测 · 校准决策 · 人工复核 · 标注回流")
base = os.getenv("QC_API_URL", "http://127.0.0.1:8000").rstrip("/")
headers = {"X-API-Key": os.getenv("QC_API_KEY", "")}


def request(method, path, **kwargs):
    try:
        response = httpx.request(method, base + path, headers=headers, timeout=60, **kwargs)
        response.raise_for_status()
        return response
    except httpx.HTTPError as exc:
        message = exc.response.text if isinstance(exc, httpx.HTTPStatusError) else str(exc)
        st.error(f"服务请求失败：{message}")
        st.stop()


info = request("GET", "/model").json()
stats = request("GET", "/stats").json()
with st.sidebar:
    st.subheader("当前模型")
    st.code(info["version"])
    st.write("类别：", info["category"])
    st.caption(info["backend"])
    if info["synthetic"]:
        st.warning("合成数据演示：指标仅用于验证流程，不代表工业性能。")
    st.write("自动放行验证：", "通过" if info["release_gate"]["approved"] else "证据不足，转人工")
    st.caption("验证结论依赖固定模型、独立样本和同分布条件。")
    if st.button("刷新数据", width="stretch"):
        st.rerun()

cols = st.columns(3)
cols[0].metric("待复核", stats.get("pending", 0))
cols[1].metric("已标注", stats.get("labeled", 0))
cols[2].metric("自动决策", stats.get("auto", 0))
tab_inspect, tab_review, tab_learn, tab_eval = st.tabs(["图像检测", "复核队列", "主动学习", "模型与验证"])
route_names = {"pass": "放行", "alarm": "报警", "review": "人工复核"}

with tab_inspect:
    category = st.text_input("产品类别", info["category"])
    uploaded = st.file_uploader("上传待检图像", type=["png", "jpg", "jpeg", "bmp", "webp"])
    if st.button("开始检测", type="primary", disabled=uploaded is None):
        with st.spinner("正在检测并生成热力图…"):
            st.session_state["last"] = request("POST", "/inspect", files={"file": (uploaded.name, uploaded.getvalue())},
                                              data={"category": category}).json()
    if "last" in st.session_state:
        row = st.session_state["last"]
        result = row["result"]
        st.subheader(route_names[result["route"]])
        a, b = st.columns(2)
        a.image(request("GET", f'/reviews/{row["id"]}/image').content, caption="原图", width="stretch")
        b.image(request("GET", f'/reviews/{row["id"]}/image?heatmap=true').content, caption="异常热力图与候选区域", width="stretch")
        st.write(result["explanation"])
        st.caption(f'路由原因：{result["reason"]} · 推理 {result["latency_ms"]:.1f} ms · 分数 {result["score"]:.3f}')
        if result.get("secondary"):
            st.info(f'实验二级分割参考分数：{result["secondary"]["score"]:.3f}；不是校准概率，仅辅助人工复核。')
        if result.get("stage_timing"):
            with st.expander("分阶段耗时"):
                st.json(result["stage_timing"])
        with st.expander("校准详情"):
            st.json({k: result[k] for k in ("prediction_set", "p_normal", "p_defect", "defect_probability", "drift")})
            st.caption("Conformal p 值不等于缺陷概率；概率估计来自单独的拟合集。")

with tab_review:
    status = st.selectbox("队列", ["pending", "auto", "labeled"], format_func=lambda x: {"pending": "待复核", "auto": "自动决策抽查", "labeled": "已标注 / 修正"}[x])
    rows = request("GET", f"/reviews?status={status}").json()
    if not rows:
        st.info("当前队列为空。上传新图像后，模糊决策和抽检样本会进入复核队列。")
    else:
        selected = st.selectbox("选择样本", rows, format_func=lambda r: f'{r["id"]} · {r["result"]["reason"]}')
        if selected["label"] is not None:
            st.caption(f'已保存结论：{"正常" if selected["label"] == 0 else "缺陷"} · 修订 {selected["revision"]}')
            with st.expander("标注历史"):
                st.dataframe(request("GET", f'/reviews/{selected["id"]}/history').json(), hide_index=True)
        left, right = st.columns([1.4, 1])
        left.image(request("GET", f'/reviews/{selected["id"]}/image?heatmap=true').content, width="stretch")
        with right, st.form(f'feedback-{selected["id"]}-{selected["revision"]}'):
            label = st.radio("人工结论", [0, 1], format_func=lambda x: "正常" if x == 0 else "缺陷", index=None)
            reviewer = st.text_input("标注人", value="operator")
            note = st.text_area("备注 / 缺陷描述")
            if st.form_submit_button("保存标注", type="primary"):
                if label is None or not reviewer.strip():
                    st.error("请选择结论并填写标注人。")
                else:
                    request("POST", f'/reviews/{selected["id"]}/feedback', json=dict(label=label, reviewer=reviewer, note=note, revision=selected["revision"]))
                    st.rerun()
        if info.get("schema_version")==2 and selected["label"]==1:
            st.caption("缺陷像素掩膜：与原图同尺寸，黑色为背景，非黑色为缺陷。先保存人工结论，再上传掩膜。")
            mask_upload=st.file_uploader("上传缺陷掩膜",type=["png","bmp"],key="mask-"+selected["id"])
            seconds=st.number_input("本次像素标注耗时（秒）",min_value=0.0,value=0.0)
            if st.button("保存像素掩膜",disabled=mask_upload is None):
                data={"revision":selected["revision"]}
                if seconds>0:
                    data["annotation_seconds"]=seconds
                request("POST",f'/reviews/{selected["id"]}/mask',files={"file":(mask_upload.name,mask_upload.getvalue())},data=data)
                st.success("像素掩膜已保存，可用于二级分割训练。")

with tab_learn:
    strategy = st.selectbox("选样策略", ["uncertainty", "coreset", "hybrid", "random"])
    budget = st.slider("本轮标注预算", 1, 50, 10)
    batch = request("GET", f"/active-learning?strategy={strategy}&budget={budget}").json()
    st.write(f'可选样本：{batch["eligible"]}；本轮建议：{len(batch["ids"])}')
    st.dataframe(pd.DataFrame({"建议复核样本 ID": batch["ids"]}), hide_index=True, width="stretch")
    st.caption("只从当前模型的未标注复核队列选样。新标签形成候选版本，经独立验证后再切换。")
    if info.get("schema_version")==2:
        st.caption("正常反馈更新一级模型；带像素掩膜的缺陷反馈训练二级分割。训练完成后保留为候选版本。")
        if st.button("训练新候选模型"):
            st.session_state["training_job"]=request("POST","/training-jobs").json()["id"]
        if "training_job" in st.session_state:
            job=request("GET","/training-jobs/"+st.session_state["training_job"]).json()
            st.json(job)
            if job["status"]=="running" and st.button("取消训练"):
                request("POST","/training-jobs/"+job["id"]+"/cancel")
                st.rerun()
    else:
        st.code(f'vision-qc retrain --run "{info["run_dir"]}" --state "{info["state_dir"]}" --output runs/candidate-01', language="bash")

with tab_eval:
    validation = info["release_gate"]["validation"]
    a, b, c = st.columns(3)
    a.metric("验证缺陷数", validation["n_defect"])
    b.metric("验收前策略漏检率", f'{validation["auto_release_fnr"]:.2%}' if validation["auto_release_fnr"] is not None else "无缺陷证据")
    c.metric("漏检率 95% 单侧上界", f'{validation["fnr_upper_95"]:.2%}')
    st.caption("这里评估风险守门前的策略；未通过验收时，实际服务关闭自动放行。人工复核准确率需单独评估。")
    if info.get("validation_reference_only"):
        st.warning("此处为父模型的历史风险结果，仅供参考。当前候选已改变完整策略，必须使用新数据重新验收。")
    with st.expander("当前模型详细信息"):
        st.json(info)
    st.subheader("评测报告")
    reports = request("GET", "/reports").json()
    if reports:
        selected_report = st.selectbox("选择运行结果", reports, format_func=lambda item: item["name"])
        report = selected_report["report"]
        st.caption("合成流程测试" if report.get("synthetic") else "公开 / 真实数据评测，请核对协议和样本数")
        r1, r2, r3 = st.columns(3)
        r1.metric("Image AUROC", f'{report["image_auroc"]:.4f}' if report.get("image_auroc") is not None else "—")
        r2.metric("测试图像数", report.get("n_test", report.get("n", "—")))
        r3.metric("缺陷图像数", report.get("n_test_defect", report.get("n_defect", "—")))
        with st.expander("协议、定位指标与完整结果"):
            st.json(report)
    else:
        st.info("运行评测命令后，结果会显示在这里。")
