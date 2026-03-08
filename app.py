import streamlit as st
import pandas as pd
import numpy as np
import os
import io
from openai import OpenAI
from utils import preprocess_signal
from model_loader import load_models, TinySTGNN_Optimized, predict_cognitive_state
import re
from reportlab.lib import colors
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle

# ==============================================================================
# 1. 基础配置
# ==============================================================================
st.set_page_config(
    page_title="VR 物理实验 · 认知负荷智能监测系统",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded"
)

# 注册中文字体 (用于PDF导出)
def register_chinese_font():
    try:
        # macOS 常见中文字体路径
        font_path = "/System/Library/Fonts/Supplemental/Arial Unicode.ttf"
        if os.path.exists(font_path):
            pdfmetrics.registerFont(TTFont('ArialUnicode', font_path))
            return 'ArialUnicode'
        else:
            return None
    except:
        return None

# 生成 PDF 报告 (专业排版版)
def create_pdf(student_grade, experiment_name, is_overload, advice_text):
    buffer = io.BytesIO()
    
    # 使用 Platypus 布局引擎，支持自动换行、分页和样式
    doc = SimpleDocTemplate(
        buffer, 
        pagesize=A4,
        rightMargin=50, leftMargin=50, 
        topMargin=50, bottomMargin=50
    )
    
    story = []
    
    # 注册字体
    font_name = register_chinese_font()
    if not font_name:
        font_name = "Helvetica" # Fallback
        
    # 定义样式
    styles = getSampleStyleSheet()
    
    # 自定义标题样式
    title_style = ParagraphStyle(
        'ReportTitle',
        parent=styles['Title'],
        fontName=font_name,
        fontSize=22,
        leading=28,
        spaceAfter=20,
        alignment=1, # Center
        textColor=colors.HexColor('#2C3E50')
    )
    
    # 自定义正文样式
    normal_style = ParagraphStyle(
        'ReportNormal',
        parent=styles['Normal'],
        fontName=font_name,
        fontSize=10.5,
        leading=18,
        spaceAfter=8,
        alignment=0 # Left
    )
    
    # 自定义小标题样式
    heading_style = ParagraphStyle(
        'ReportHeading',
        parent=styles['Heading2'],
        fontName=font_name,
        fontSize=14,
        leading=20,
        spaceBefore=15,
        spaceAfter=10,
        textColor=colors.HexColor('#34495E'),
        borderPadding=0
    )
    
    # 状态强调样式
    status_style = ParagraphStyle(
        'StatusStyle',
        parent=normal_style,
        fontName=font_name,
        textColor=colors.red if is_overload else colors.green,
        fontSize=10.5,
        leading=18
    )

    # --- 构建文档内容 ---
    
    # 1. 报告标题
    story.append(Paragraph("认知负荷干预建议报告", title_style))
    story.append(Spacer(1, 10))
    
    # 2. 实验基本信息 (使用表格对齐)
    status_text = "认知超载 (High Load)" if is_overload else "心流状态 (Flow State)"
    
    # 表格数据
    info_data = [
        ["实验名称:", experiment_name],
        ["学生学段:", student_grade],
        ["检测状态:", Paragraph(status_text, status_style)] # 在表格中嵌入 Paragraph 以支持颜色
    ]
    
    t = Table(info_data, colWidths=[70, 400])
    t.setStyle(TableStyle([
        ('FONTNAME', (0,0), (-1,-1), font_name),
        ('FONTSIZE', (0,0), (-1,-1), 10.5),
        ('TEXTCOLOR', (0,0), (0,-1), colors.gray), # 标签列为灰色
        ('VALIGN', (0,0), (-1,-1), 'TOP'),
        ('BOTTOMPADDING', (0,0), (-1,-1), 6),
    ]))
    story.append(t)
    
    story.append(Spacer(1, 15))
    # 分割线
    story.append(Paragraph("_" * 80, normal_style))
    story.append(Spacer(1, 15))
    
    # 3. 建议详情内容
    story.append(Paragraph("干预建议详情:", heading_style))
    
    # 解析 Markdown 文本并转换为 ReportLab 元素
    lines = advice_text.split('\n')
    for line in lines:
        line = line.strip()
        if not line:
            continue
            
        # 处理粗体: **text** -> <b>text</b>
        formatted_line = re.sub(r'\*\*(.*?)\*\*', r'<b>\1</b>', line)
        
        if line.startswith('###') or line.startswith('####'):
            # 标题
            clean_text = line.lstrip('#').strip()
            story.append(Paragraph(clean_text, heading_style))
            
        elif line.startswith('- ') or line.startswith('* '):
            # 列表项
            clean_text = line[2:].strip()
            clean_text = re.sub(r'\*\*(.*?)\*\*', r'<b>\1</b>', clean_text) # 再次处理粗体
            
            bullet_style = ParagraphStyle(
                'ReportBullet',
                parent=normal_style,
                leftIndent=15,
                bulletIndent=0,
                firstLineIndent=0
            )
            story.append(Paragraph(f"• {clean_text}", bullet_style))
            
        elif line == '---':
             # 分割线
             story.append(Spacer(1, 5))
             story.append(Paragraph("_" * 80, normal_style))
             story.append(Spacer(1, 5))
        else:
            # 普通段落
            story.append(Paragraph(formatted_line, normal_style))
            
    # 生成 PDF
    doc.build(story)
    buffer.seek(0)
    return buffer

# DeepSeek API 配置 (生产环境请使用 st.secrets)
# 安全提示：请勿将真实 API Key 直接提交到 GitHub
try:
    API_KEY = st.secrets["DEEPSEEK_API_KEY"]
except:
    API_KEY = "sk-xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx" # 占位符，请在本地 .streamlit/secrets.toml 中配置
    
BASE_URL = "https://api.deepseek.com"

# 初始化 OpenAI 客户端
client = OpenAI(api_key=API_KEY, base_url=BASE_URL)

# ==============================================================================
# 2. 辅助函数
# ==============================================================================
@st.cache_resource
def get_models():
    """缓存加载模型，避免重复加载"""
    return load_models('.')

@st.cache_data
def load_data_file(uploaded_file):
    """
    读取上传的文件，增加缓存以提升性能。
    如果是 Excel 且文件过大，可能会导致内存问题，建议使用 CSV。
    """
    try:
        if uploaded_file.name.endswith('.csv'):
            return pd.read_csv(uploaded_file)
        else:
            # Excel 读取比较消耗内存，增加 engine='openpyxl'
            return pd.read_excel(uploaded_file, engine='openpyxl')
    except Exception as e:
        st.error(f"读取文件失败: {str(e)}")
        return None

def generate_advice_cn(grade, experiment_name, is_overload):
    """调用 DeepSeek API 生成中文教学建议"""
    state_desc = "认知超载 (高脑力负荷)" if is_overload else "心流状态 (最佳认知负荷)"
    
    prompt = f"""
    你是一位资深的教育心理学专家和一线教师。
    场景：一名【{grade}】的学生正在进行名为“{experiment_name}”的 VR 实验。
    实时监测：系统检测到该学生当前处于【{state_desc}】。
    
    任务：请为指导教师提供 3 条具体的、可操作的干预建议。
    - 如果是“认知超载”：重点在于如何降低任务难度、提供支架或暂停引导。
    - 如果是“心流状态”：重点在于如何保持学生专注，或适当增加挑战以维持投入。
    
    输出格式要求：
    1. **状态解读**：用通俗语言解释当前学生的状态。
    2. **即时干预**：教师现在立刻应该做什么。
    3. **后续策略**：接下来的实验环节如何调整。
    
    请用中文回答，语气专业且亲切。
    """
    
    try:
        response = client.chat.completions.create(
            model="deepseek-chat",
            messages=[
                {"role": "system", "content": "你是一位专业的教育辅助AI助手。"},
                {"role": "user", "content": prompt}
            ],
            temperature=0.7,
            max_tokens=800
        )
        return response.choices[0].message.content
    except Exception as e:
        return f"AI 建议生成失败: {str(e)}"

# ==============================================================================
# 3. 主界面逻辑
# ==============================================================================
def main():
    # ------------------------------------------------------------------------------
    # 标题
    # ------------------------------------------------------------------------------
    st.title("🧠 VR 物理实验 · 认知负荷智能监测系统")
    st.markdown("### 基于 ST-GNN 与梯度提升树的生理信号实时分析平台")
    
    # 侧边栏：配置
    with st.sidebar:
        st.header("⚙️ 实验配置")
        experiment_name = st.text_input("实验名称", "物理实验：光的折射")
        student_grade = st.selectbox("学生学段", ["初中", "高中", "大学/大专", "成人教育"])
        
        st.markdown("---")
        st.markdown("**系统信息**")
        st.caption("模型架构: ST-GNN + Gradient Boosting")
        st.caption("当前版本: v1.0.0 (CN)")
        
        # 提供样本下载链接 (模拟)
        with open("sample_data.csv", "rb") as f:
            st.download_button(
                label="📥 下载测试样本数据 (CSV)",
                data=f,
                file_name="sample_data.csv",
                mime="text/csv"
            )

    # 主内容区
    # --------------------------------------------------------------------------
    # 1. 顶部：数据上传区 (全宽布局，更清晰)
    # --------------------------------------------------------------------------
    with st.container():
        st.subheader("📂 步骤 1：导入生理数据")
        st.markdown("请上传包含学生生理信号的 Excel 或 CSV 文件以启动分析。")
        
        # 使用列布局来放置提示信息和上传组件，使其更紧凑
        uc1, uc2 = st.columns([3, 1])
        with uc1:
            uploaded_file = st.file_uploader("点击下方区域上传文件", type=['csv', 'xlsx'], help="支持拖拽上传")
        with uc2:
            st.info("💡 **推荐格式**\n\n使用 **CSV** 文件\n包含 `PPG`, `EMG` 等5通道")

    # --------------------------------------------------------------------------
    # 2. 分析与结果展示区
    # --------------------------------------------------------------------------
    if uploaded_file:
        st.divider() # 添加分割线
        
        # 读取文件
        with st.spinner("🔄 正在解析数据文件..."):
            try:
                df = load_data_file(uploaded_file)
            except Exception as e:
                st.error(f"严重错误：文件读取失败。\n详细信息: {e}")
                df = None
        
        if df is not None:
            # 校验与预处理
            required = ['PPG', 'EMG', 'EEG', 'SCR', 'ECG']
            missing = [col for col in required if col not in df.columns]
            
            if missing:
                st.error(f"❌ 数据格式错误：缺少必要通道 {missing}")
            else:
                try:
                    # 预处理与推理
                    with st.spinner("🧠 正在进行多模态特征融合与模型推理..."):
                        input_tensor = preprocess_signal(df) 
                        stgnn, gb_clf, scaler = get_models()
                        
                        if stgnn is None:
                            st.error("❌ 模型加载失败，请联系管理员。")
                            st.stop()
                            
                        pred, proba = predict_cognitive_state(stgnn, gb_clf, scaler, input_tensor)
                        is_overload = (pred == 1)
                        confidence = proba[pred]

                    # --- 结果仪表盘 ---
                    st.subheader("📊 步骤 2：实时认知状态分析")
                    
                    # 使用卡片式布局展示核心指标
                    res_col1, res_col2, res_col3 = st.columns([2, 2, 3])
                    
                    with res_col1:
                        # 状态指标
                        if is_overload:
                            st.error("⚠️ **认知超载**")
                            st.metric("当前状态", "High Load", "需干预", delta_color="inverse")
                        else:
                            st.success("✅ **心流状态**")
                            st.metric("当前状态", "Flow State", "状态佳", delta_color="normal")
                            
                    with res_col2:
                        # 置信度指标
                        st.write("**模型置信度**")
                        st.progress(float(confidence), text=f"{confidence:.1%}")
                        st.caption(f"基于 {len(df)} 条时序数据样本分析")

                    with res_col3:
                        # 简要解读
                        st.info(
                            "**分析摘要**：\n\n"
                            f"模型检测到学生在实验过程中处于 **{'高负荷' if is_overload else '最佳'}** 认知区间。"
                            f"{'建议立刻采取教学干预措施以降低挫败感。' if is_overload else '建议维持当前教学节奏，引导学生深入探索。'}"
                        )

                    # --- AI 建议区 ---
                    st.divider()
                    st.subheader("🤖 步骤 3：AI 智能教学助手")
                    
                    # 紧凑布局：按钮紧跟标题下方
                    if st.button("✨ 生成干预建议", type="primary"):
                        st.session_state['generate_advice'] = True
                    
                    if st.session_state.get('generate_advice'):
                            # 检查 API Key 是否配置
                            if API_KEY.startswith("sk-xxxx"):
                                st.error("⚠️ 未配置 API Key。请在本地 `.streamlit/secrets.toml` 文件中配置 `DEEPSEEK_API_KEY`。")
                            else:
                                with st.chat_message("assistant", avatar="🎓"):
                                    if 'advice_content' not in st.session_state:
                                        with st.spinner("正在咨询 DeepSeek 教育心理学专家..."):
                                            advice = generate_advice_cn(student_grade, experiment_name, is_overload)
                                            st.session_state['advice_content'] = advice
                                    
                                    st.markdown(st.session_state['advice_content'])
                                    
                                    # PDF 导出按钮
                                    pdf_buffer = create_pdf(student_grade, experiment_name, is_overload, st.session_state['advice_content'])
                                    st.download_button(
                                        label="📄 导出建议报告 (PDF)",
                                        data=pdf_buffer,
                                        file_name="intervention_advice.pdf",
                                        mime="application/pdf"
                                    )
                    else:
                        st.info("点击上方按钮，系统将结合**学生学段**与**当前认知状态**，为您生成个性化的教学干预策略。")

                except Exception as e:
                    st.error(f"分析过程发生错误: {str(e)}")
                    with st.expander("查看详细错误日志"):
                        st.write(e)

    with st.expander("🛠️ 调试与元数据"):
        st.write("工作路径:", os.getcwd())
        st.write("上传文件:", uploaded_file.name if uploaded_file else "无")

if __name__ == "__main__":
    main()
