# Lecture 01 — 计算机与开发环境

> 在开始写代码之前，这门课带你理解并搭建好必不可少的"工作环境"。
> 从计算机的工作原理讲起，一路覆盖终端、Python 安装、Git、Docker 和云端 GPU —
> 学习 AI 所需的全部基本功都在这里打下。

## 这门课你将学到

- 直观理解计算机 (CPU、内存、存储设备) 实际上在做什么
- 文件、文件夹、路径、终端命令这些开发者的"基本语言"
- 安装 Python、配置虚拟环境、编辑器和 Jupyter 等实战工具
- 用 Git/GitHub 安全地管理工作成果并与他人协作
- 环境变量、密钥、Docker、云端 GPU 等实际工作环境的感觉

## 先修课程

无。这是整个课程体系的第一门课。只要有用过 Excel 这种程度的计算机经验就足够了。

## 关卡目录

| 关卡 | 标题 | 难度 |
|---|---|---|
| [level00](level00_what_computers_do/README.md) | 计算机到底是一台做什么的机器 | ⭐ |
| [level01](level01_files_folders_paths/README.md) | 文件、文件夹与路径的概念 | ⭐ |
| [level02](level02_terminal_basics/README.md) | 终端第一步 | ⭐ |
| [level03](level03_install_python/README.md) | 安装 Python 与第一次运行 | ⭐ |
| [level04](level04_venv_packages/README.md) | 虚拟环境与包管理 | ⭐⭐ |
| [level05](level05_editors_jupyter/README.md) | 代码编辑器与 Jupyter | ⭐⭐ |
| [level06](level06_git_basics/README.md) | Git — 版本管理的概念 | ⭐⭐ |
| [level07](level07_github_collaboration/README.md) | GitHub — 远程仓库与协作 | ⭐⭐ |
| [level08](level08_shell_automation/README.md) | 用 Shell 脚本自动化重复工作 | ⭐⭐⭐ |
| [level09](level09_env_vars_secrets/README.md) | 安全地管理环境变量与密钥 | ⭐⭐⭐ |
| [level10](level10_docker_reproducibility/README.md) | Docker — 把整个环境原样复现 | ⭐⭐⭐⭐ |
| [level11](level11_cloud_gpu_servers/README.md) | 远程服务器、云端与 GPU 环境 | ⭐⭐⭐⭐ |

## 快速通道 (时间紧就只看这些)

1. **level02 终端第一步** — 之后所有课程的实战都从终端开始。
2. **level03 安装 Python 与第一次运行** — 亲手敲下 `python3 main.py` 的那一刻就是起点。
3. **level04 虚拟环境与包管理** — 工作中一半的"在我电脑上明明能跑"事故都能在这里预防。
4. **level06 Git 基础** — 保证工作成果不丢失的最低限度的安全装置。
5. **level09 环境变量与密钥** — 防止 API 密钥泄露的习惯必须从一开始就养成。

## 这门课在实际工作中的应用场景

- **报表自动化的第一步**: 能用一个脚本搞定每周重复的文件整理和重命名的同事，就是理解了这门课 level08 的人。
- **预防协作事故**: 不再用"最终版_真最终版_修改2.xlsx"，而是用 Git 提交历史管理文档，谁在什么时候改了什么，1 分钟内就能查到。
- **通过安全审计**: 不在代码里硬编码密钥的习惯 (level09)，是公司内部安全检查和外部审计最先核查的项目。
- **AI 项目预算会议**: 明白 GPU 云服务为什么按小时计费 (level11)，在和开发团队开预算会时才能对得上话。

## 实战方法

在每个关卡文件夹里输入下面这一行即可。完全不需要联网或 API 密钥。

```bash
python3 main.py
```
