# 🌐 保姆级教程：如何将 Python 网页发布到网上

这个教程会手把手教你如何将这个项目发布到 **Streamlit Community Cloud**。发布后，你会获得一个网址（URL），发给任何人都能直接用，不需要他们下载代码。

---

## 第一阶段：准备工作 (在你的电脑上)

1. **检查文件**：
   确保 `cognitive_alert_system_packaged` 文件夹里有以下关键文件（我已经帮你准备好了）：
   - `app.py` (主程序)
   - `requirements.txt` (告诉服务器需要安装什么库)
   - `packages.txt` (如果需要系统级依赖，目前不需要)
   - 其他模型文件 (`.pth`, `.pkl`, `.csv` 等)

2. **注册 GitHub 账号** (如果你还没有)：
   - 去 [github.com](https://github.com) 注册一个免费账号。
   - 记住你的用户名和密码。

---

## 第二阶段：上传代码到 GitHub

Streamlit Cloud 需要从 GitHub 读取你的代码。

### 简单上传法 (网页版)：
1. 登录 GitHub，点击右上角的 **+** 号，选择 **New repository** (新建仓库)。
2. **Repository name** 填一个名字，比如 `vr-cognitive-monitor`。
3. 选中 **Public** (公开) 或 **Private** (私有) 都可以（Streamlit Cloud 都支持）。
4. 勾选 **Add a README file**。
5. 点击 **Create repository**。
6. 在新页面点击 **Add file** -> **Upload files**。
7. 将 `cognitive_alert_system_packaged` 文件夹里的**所有文件**（注意是文件夹里面的内容，不是文件夹本身）拖进去。
   - 也就是 `app.py`, `requirements.txt`, `model_loader.py`, 以及所有模型文件等。
8. 等待文件上传完毕，在下方 "Commit changes" 处点击绿色按钮 **Commit changes**。

---

## 第三阶段：一键部署到 Streamlit Cloud

1. **注册/登录 Streamlit**：
   - 访问 [share.streamlit.io](https://share.streamlit.io/)。
   - 点击 **Continue with GitHub** (用 GitHub 账号登录)。
   - 授权 Streamlit 访问你的 GitHub 仓库。

2. **开始部署**：
   - 登录后，点击右上角的 **New app** (新建应用)。
   - **Repository**: 选择你刚才创建的 GitHub 仓库 (例如 `yourname/vr-cognitive-monitor`)。
   - **Branch**: 默认选 `main` (或 `master`)。
   - **Main file path**: 填 `app.py` (如果它已经自动识别出来了就不用管)。
   - 点击 **Deploy!** (部署)。

3. **等待安装**：
   - 你会看到一个黑底白字的终端界面，显示 "Running..."。
   - 它正在云端服务器安装 `requirements.txt` 里的库（PyTorch, Pandas 等）。
   - 这可能需要 **3-5 分钟**，请耐心等待。
   - 安装完成后，网页会自动刷新，显示你的应用界面！

---

## 第四阶段：配置 API Key (重要！)

因为你的代码里用到了 DeepSeek 的 API 来生成建议，你需要告诉云端服务器你的 API Key 是什么。**千万不要把 Key 直接写在代码里上传到 GitHub！**

1. 在你的应用界面右下角，点击 **Manage app** (管理应用)。
2. 点击 **Settings** (三个点或者齿轮图标)。
3. 点击 **Secrets** 标签页。
4. 在大文本框里，按照下面的格式粘贴你的 API Key：

```toml
DEEPSEEK_API_KEY = "sk-你的真实API密钥在这里"
```

5. 点击 **Save**。
6. 刷新你的网页应用，现在 AI 建议功能就可以正常使用了！

---

## 常见问题 (Troubleshooting)

- **Q: 部署时报错 "ModuleNotFoundError"?**
  - A: 检查 `requirements.txt` 是否在 GitHub 仓库的根目录下。

- **Q: 报错 "Out of memory" (内存不足)?**
  - A: 免费版 Streamlit Cloud 有 1GB 内存限制。如果模型太大可能会崩。目前的 ST-GNN 模型很小，应该没问题。如果遇到，请尝试上传更小的 Excel/CSV 文件。

- **Q: 只有我自己能看吗？**
  - A: 如果你的 GitHub 仓库是 Public (公开) 的，任何人有链接都能看。如果是 Private (私有) 的，你需要点击右上角的 "Share" 邀请别人的邮箱才能看。

---

## 搞定！🎉
现在，你可以把浏览器地址栏的链接复制下来，发给你的用户。他们打开就能用，完全不需要安装 Python！
