# 🧠 VR 物理实验 · 认知负荷智能监测系统

基于多模态生理信号（PPG, EMG, EEG, SCR, ECG）与混合深度学习模型（ST-GNN + Gradient Boosting）的 VR 教学认知负荷实时监测与干预建议系统。

## 🌟 项目简介

本项目旨在解决 VR 物理实验教学中学生认知负荷过高导致学习效率下降的问题。通过采集学生在实验过程中的多模态生理信号，系统能够实时识别学生的认知状态（心流/超载），并结合 DeepSeek 大模型生成个性化的教学干预建议。

### 核心功能
1.  **多模态数据分析**：支持 PPG、EMG、EEG、SCR、ECG 五种生理信号的自动处理与特征提取。
2.  **混合智能推理**：
    *   **ST-GNN (时空图神经网络)**：捕捉多模态信号间的时空依赖关系。
    *   **Gradient Boosting (梯度提升树)**：针对小样本数据进行鲁棒分类。
3.  **实时仪表盘**：可视化展示各生理指标（心率、皮电、脑电波段等）的实时变化。
4.  **AI 教学助手**：集成 DeepSeek R1 大模型，根据学生的认知状态和实验内容，生成针对性的教学引导策略，并支持导出专业 PDF 报告。

## �️ 技术栈

*   **前端框架**: [Streamlit](https://streamlit.io/)
*   **深度学习**: PyTorch (ST-GNN 实现)
*   **机器学习**: Scikit-learn (Gradient Boosting, Random Forest)
*   **数据处理**: NumPy, Pandas, SciPy
*   **大模型集成**: DeepSeek API (OpenAI SDK 兼容)
*   **PDF 生成**: ReportLab (Platypus 引擎)

## � 快速开始

### 1. 环境准备

推荐使用 Python 3.9+ 环境。

```bash
# 克隆项目
git clone https://github.com/your-username/vr-cognitive-load-monitor.git
cd vr-cognitive-load-monitor

# 安装依赖
pip install -r requirements.txt
```

### 2. 配置 API Key

本项目使用 DeepSeek API 生成教学建议。为了安全起见，请勿直接在代码中硬编码 API Key。

在项目根目录下创建 `.streamlit/secrets.toml` 文件：

```toml
# .streamlit/secrets.toml
DEEPSEEK_API_KEY = "sk-your-api-key-here"
```

> 如果没有 API Key，系统将无法生成 AI 建议，但监测与仪表盘功能仍可正常使用。

### 3. 运行系统

```bash
streamlit run app.py
```

启动后，浏览器将自动打开 `http://localhost:8501`。

## 📂 数据格式说明

系统支持上传 `.csv` 或 `.xlsx` 格式的生理信号数据。数据应包含以下列（大小写敏感）：

| 列名 | 说明 | 单位/备注 |
| :--- | :--- | :--- |
| `Time` | 时间戳 | 秒 (s) |
| `PPG` | 光电容积脉搏波 | - |
| `EMG` | 肌电信号 | - |
| `EEG` | 脑电信号 | - |
| `SCR` | 皮肤电反应 | - |
| `ECG` | 心电信号 | - |

> **注意**：系统会自动验证数据长度，建议上传时长不低于 30 秒的数据片段。

## 🧬 模型架构细节

本系统采用两阶段混合推理架构：

1.  **特征提取层**: 使用预训练的 **ST-GNN** (Spatio-Temporal Graph Neural Network) 从原始多模态时间序列中提取高维时空特征向量。
2.  **决策层**: 将提取的特征输入 **Gradient Boosting Classifier**，结合统计特征（如 HRV、Alpha/Beta 波比值等）进行最终的二分类（High Load / Low Load）。

## � 许可证

[MIT License](LICENSE)
