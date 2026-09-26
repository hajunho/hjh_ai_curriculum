# hjh AI Curriculum — 写给非科班职场人的 AI 通关课程(简体中文版)

> **从只会用 Excel 的人，一路学到机器学习、深度学习和 LLM 工程。**
> 这是一套为没有 IT 专业背景的职场人设计的实战型课程，共 12 门课、144 个关卡，
> 目标是让你成为“真正理解 AI、并能亲手使用 AI 的人”。

- **作者**: 河俊镐 (hajunho) · ZeliDesk / 教育科技系
- **关于本文件夹**: 这是简体中文版 (Simplified Chinese edition)，韩语原版位于仓库根目录。
  数据生成器 (`zh/common/hjh_data.py`) 也已完整本地化为中文，
  只用这个 `zh/` 文件夹就可以完成全部学习。
- **许可证**: MIT — 所有讲义和代码都是为本课程全新创作的原创内容，
  没有复制任何外部教材或数据集，练习数据也全部由代码直接生成。
- **语言**: 讲义为简体中文，代码标识符为英文

---

## 这套课程的不同之处

1. **先讲比喻，后讲公式** — 每个概念都先用工作和日常生活中的类比来解释。
2. **没有安装地狱** — 练习数据由代码直接生成，不需要任何下载，
   全部课程在没有 GPU 的普通笔记本电脑上都能跑完。
3. **用业务问题来学习** — 销售额预测、客户流失、异常交易检测、公司内部文档问答机器人等，
   全都是你在公司里真正会遇到的问题。
4. **包含从零实现** — 感知机、反向传播、分词器，甚至迷你 GPT，都亲手实现一遍，
   把“黑箱”彻底打开看个明白。

## 整体结构

共 12 门课 (lecture01~12)，每门课由 12 个关卡 (level00~11) 组成。
level00 是完全零基础入门 (比喻和概念)，level11 则是实战进阶。

| 课程 | 主题 | 一句话介绍 |
|---|---|---|
| [lecture01](lecture01_computer_and_environment/) | 计算机与开发环境 | 从终端、Python 安装、Git 到云端 |
| [lecture02](lecture02_python_basics/) | Python 编程基础 | 从变量到面向对象和测试 |
| [lecture03](lecture03_data_handling/) | 数据处理 | 跳出 Excel — NumPy · Pandas |
| [lecture04](lecture04_database_sql/) | 数据库与 SQL | 从 SELECT 到窗口函数和 Python 连接 |
| [lecture05](lecture05_statistics_visualization/) | 统计与可视化 | 从平均数的陷阱到 A/B 测试 |
| [lecture06](lecture06_machine_learning_basics/) | 机器学习入门 | 回归、分类、集成学习、模型解释 |
| [lecture07](lecture07_business_ml_practice/) | 商业机器学习实战 | 销售额预测、客户流失、异常交易检测 |
| [lecture08](lecture08_deep_learning_foundations/) | 深度学习基础 | 从零实现感知机，一直到 PyTorch |
| [lecture09](lecture09_computer_vision/) | 计算机视觉 | CNN、迁移学习、工业应用 |
| [lecture10](lecture10_nlp_text/) | 自然语言处理 | 从中文文本预处理到亲手实现 Transformer |
| [lecture11](lecture11_llm_and_rag/) | LLM 应用与 RAG | 提示词、嵌入、公司内部文档问答机器人 |
| [lecture12](lecture12_llm_engineering_mlops/) | LLM 工程与 MLOps | 分词器、迷你 GPT、对齐、量化、部署 |

详细目录见 [CURRICULUM.md](CURRICULUM.md)，环境准备见 [SETUP.md](SETUP.md)，
遇到不懂的术语请查 [GLOSSARY.md](GLOSSARY.md)。

## 怎么学

1. 按 `SETUP.md` 准备好 Python 环境 (30 分钟)。
2. 进入每个关卡文件夹，先读 `README.md` → 再运行 `main.py` → 最后做练习题。
3. 每个关卡需要 30~90 分钟。每周完成 5 个关卡，**大约 7 个月**即可通关。
4. 按顺序学是标准路线，但每门课的 README 里也提供了“快速路线”。

```bash
git clone https://github.com/hajunho/hjh_ai_curriculum.git
cd hjh_ai_curriculum/zh
python3 -m venv .venv && source .venv/bin/activate
pip install -r ../requirements.txt
python3 lecture01_computer_and_environment/level00_what_computers_do/main.py
```

## 版权与许可证

本仓库中的所有文档、代码和数据生成器，都是河俊镐 (hajunho) 为这套课程全新编写的。
没有搬运任何商业培训课程或书籍的内容，也不包含任何外部数据集文件。
基于 MIT 许可证，任何人都可以自由使用、修改和再分发。
如果你在教学中使用这套课程，恳请注明出处，不胜感激。
