# 部署与打包指南 (Deployment & Packaging Guide)

这个文件夹包含了可以直接部署或打包成可执行文件（.exe/app）的所有必要文件。

## 方案一：发布为网页应用（推荐）

最简单的方法是使用 **Streamlit Community Cloud**。这是免费的，并且可以直接连接到你的 GitHub 仓库。

### 步骤：
1. **上传到 GitHub**：
   - 确保这个文件夹（`cognitive_alert_system_packaged`）里的内容已经上传到了你的 GitHub 仓库。
   - 确保 `requirements.txt` 存在（已包含）。

2. **配置 Streamlit Cloud**：
   - 访问 [Streamlit Community Cloud](https://streamlit.io/cloud) 并使用 GitHub 账号登录。
   - 点击 "New app"。
   - 选择你的 GitHub 仓库、分支（通常是 `main`）和主文件路径（`app.py`）。
   - 点击 "Deploy"。

3. **配置 Secrets (API Key)**：
   - 部署后，如果应用报错提示缺少 API Key，你需要在 Streamlit Cloud 的仪表盘中配置。
   - 点击右下角的 "Manage app" -> "Settings" -> "Secrets"。
   - 添加如下内容：
     ```toml
     DEEPSEEK_API_KEY = "sk-your-key-here"
     ```

### 优势：
- 用户无需下载任何东西，通过浏览器即可访问。
- 更新代码后，网页自动更新。
- 手机、平板都能用。

---

## 方案二：打包成可执行文件 (.exe)

如果你希望用户在没有安装 Python 的电脑上离线运行，可以将其打包成 `.exe` 文件。

### 准备工作：
1. 确保你本地已经安装了 Python 和项目依赖：
   ```bash
   pip install -r requirements.txt
   pip install pyinstaller
   ```

2. **注意**：
   - 如果你在 Windows 上运行打包脚本，会生成 Windows `.exe`。
   - 如果你在 Mac 上运行，会生成 Mac `.app` 或可执行文件。
   - **无法在 Mac 上直接生成 Windows `.exe`**（除非使用复杂的交叉编译工具）。

### 打包步骤：

1. **运行打包脚本**：
   在终端中进入当前目录，运行：
   ```bash
   python setup_exe.py
   ```
   或者直接运行：
   ```bash
   pyinstaller --onefile --additional-hooks-dir=. run_app.py --clean
   ```
   *(注意：`setup_exe.py` 已经为你配置好了所有参数，推荐直接使用它)*

2. **等待完成**：
   打包过程可能需要几分钟。完成后，你会看到一个 `dist` 文件夹。

3. **分发**：
   - 打开 `dist` 文件夹，里面会有一个 `CognitiveAlertSystem` (或 `.exe`) 文件。
   - 将这个文件发送给用户即可。
   - **注意**：由于包含了 Python 环境和 AI 模型库（如 PyTorch），文件体积可能会很大（几百 MB）。

### 常见问题：
- **文件太大？** 这是因为打包了整个 PyTorch 和 Python 环境。无法避免，除非使用更轻量的库（如 ONNX Runtime）替换 PyTorch。
- **运行报错？** 请检查终端输出。如果提示找不到文件，可能需要修改 `setup_exe.py` 中的 `datas` 列表。
- **杀毒软件误报？** 未签名的 `.exe` 可能会被 Windows Defender 拦截，属于正常现象，需点击“仍要运行”。

---

## 快速运行脚本 (本地测试)
如果你只想在本地快速启动，可以使用以下脚本：
- Windows: 双击 `run_locally.bat`
- Mac/Linux: 运行 `bash run_locally.sh`

## 文件说明
- `app.py`: 主程序代码 (已优化路径处理)。
- `run_app.py`: 用于打包的启动脚本（它会调用 Streamlit 运行 `app.py`）。
- `setup_exe.py`: 用于自动化打包的脚本。
- `requirements.txt`: 依赖列表。
- `run_locally.bat`: Windows 启动脚本。
- `run_locally.sh`: Mac/Linux 启动脚本。
