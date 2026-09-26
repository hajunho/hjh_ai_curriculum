# 环境准备指南 (SETUP)

本课程的全部内容都可以在**没有 GPU 的普通笔记本电脑**(Mac / Windows)上运行。

## 1. 确认 Python 已安装

打开终端(Mac: 终端 App，Windows: PowerShell)，输入:

```bash
python3 --version   # 3.10 以上即可 (Windows 用 python --version)
```

如果没有安装，请到 https://www.python.org/downloads/ 下载安装。
详细步骤在 lecture01/level03 里有图文并茂的讲解。

## 2. 获取仓库并创建虚拟环境

```bash
git clone https://github.com/hajunho/hjh_ai_curriculum.git
cd hjh_ai_curriculum/zh

python3 -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r ../requirements.txt
```

当提示符前面出现 `(.venv)` 就说明成功了。虚拟环境是什么，
会在 lecture01/level04 里学到。现在只需要把它理解成
“这个项目专用的工具箱”就行。

## 3. 验证是否正常工作

```bash
python3 common/hjh_data.py
```

如果输出“所有生成器均正常工作。”，准备工作就完成了。

## 4. 课程的运行方式

所有关卡的用法都一样。

```bash
cd lecture06_machine_learning_basics/level03_linear_regression
cat README.md      # 阅读讲义 (在 GitHub 上看也可以)
python3 main.py    # 运行实战代码
```

## 5. 常见问题

- **需要联网吗？** 安装完成之后就不需要了。练习数据全部由代码直接生成。
- **需要 GPU 吗？** 不需要。深度学习课程也刻意设计得很小，用 CPU 几分钟内就能跑完。
- **需要 LLM API 密钥吗？** lecture11 的部分关卡有密钥体验更好，
  但每个关卡都内置了无需密钥也能运行的离线模式。
- **报错了怎么办？** 请先查看每个关卡 README 底部的“常见错误”部分。
