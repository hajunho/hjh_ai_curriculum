"""
Lecture 01 / Level 09 — 安全地管理环境变量与密钥

实战三样把秘密放在代码之外的工具。
1) 用 os.environ 读环境变量, 2) 秘密值的掩码输出,
3) .env 文件解析器 (dotenv 原理), 4) 抓出硬编码秘密的迷你扫描器。
所有密钥值都是实战用的假值，不联网。
"""

import os
import re

# 硬编码秘密的检测规则: (说明, 正则表达式) — 真实安全工具的缩小版
SECRET_PATTERNS = [
    ("API 密钥长相 (sk-...)", re.compile(r"sk-[A-Za-z0-9]{8,}")),
    ("password= 硬编码", re.compile(r"password\s*=\s*['\"][^'\"]+['\"]", re.IGNORECASE)),
    ("secret/token 硬编码", re.compile(r"(secret|token)\s*=\s*['\"][^'\"]{6,}['\"]", re.IGNORECASE)),
]

# [4] 要检查的示例代码 — 坏例子 (秘密在代码里) 和好例子 (只引用名字)
BAD_CODE = '''
# bad_app.py — 千万别这么写!
api_key = "sk-demo1234567890abcdef"
password = "corp!2024"
db = connect("10.0.0.7", password=password)
'''
GOOD_CODE = '''
# good_app.py — 秘密只按名字引用
import os
api_key = os.environ["LLM_API_KEY"]      # 值放在环境变量(储物柜)里
password = os.environ.get("DB_PASSWORD")  # 代码里只留名牌
'''


def mask(value, show=4):
    """把秘密值遮成 'sk-d****5678' 的样子。能确认、但不暴露。"""
    if value is None:
        return "(未设置)"
    if len(value) <= show * 2:
        return "*" * len(value)
    return value[:show] + "*" * 4 + value[-show:]


def parse_dotenv(text):
    """[3] .env 解析器: 把'名字=值'格式的文本变成字典。dotenv 工具的原理。"""
    result = {}
    for line in text.splitlines():
        line = line.strip()
        if not line or line.startswith("#"):   # 跳过空行和注释
            continue
        name, _, value = line.partition("=")
        result[name.strip()] = value.strip().strip('"').strip("'")
    return result


def scan_for_secrets(code, filename):
    """[4] 迷你扫描器: 在代码里找出长着'秘密的模样'的字符串。"""
    findings = []
    for lineno, line in enumerate(code.splitlines(), start=1):
        for label, pattern in SECRET_PATTERNS:
            if pattern.search(line):
                findings.append((filename, lineno, label, line.strip()))
    return findings


def main():
    print("=" * 60)
    print("环境变量与密钥 — 秘密放代码外，代码里只留名字")
    print("=" * 60)

    # [1] 读环境变量: 给本进程设置实战用变量并读取
    os.environ["DEMO_LLM_API_KEY"] = "sk-demo1122334455667788"  # 实战用假值
    print("\n[1] 读环境变量 — 操作系统递来的'名字=值'便签")
    print(f"    os.environ['DEMO_LLM_API_KEY'] -> (读取成功, 值在下面打码展示)")
    print(f"    .get() 安全读法: 不存在的变量 -> {os.environ.get('NO_SUCH_VAR')}")
    print(f"    早就在用的环境变量 PATH 的长度: {len(os.environ.get('PATH', ''))}个字符")
    practice = os.environ.get("PRACTICE_KEY")
    print(f"    PRACTICE_KEY: {mask(practice)}  (请在'动手试试'第 1 题里 export 一下)")

    # [2] 掩码输出: 只展示'设没设'，值本身遮起来
    print("\n[2] 掩码输出 — 不把秘密整个打进日志和屏幕")
    secret = os.environ["DEMO_LLM_API_KEY"]
    print(f"    按原值直接输出      : (绝对禁止!)")
    print(f"    打码后输出          : {mask(secret)}")

    # [3] .env 解析器: 把文件内容抬进环境变量的原理
    print("\n[3] .env 模式 — 项目文件夹里的秘密保管文件 (绝对不上 Git)")
    dotenv_text = '# 实战用 .env 内容\nDB_PASSWORD="s3cret!pw"\nSLACK_TOKEN=xoxb-demo-9988\n'
    loaded = parse_dotenv(dotenv_text)
    for name, value in loaded.items():
        os.environ[name] = value           # 升格为环境变量 (dotenv 干的活)
        print(f"    {name:<14} = {mask(value)}  -> 抬进 os.environ")
    print("    -> 现在代码只要喊一声 os.environ['DB_PASSWORD'] 就行")

    # [4] 硬编码扫描器: 只有坏代码才该拉响警报
    print("\n[4] 硬编码检测迷你扫描器 — 提交前自动检查的原理")
    for filename, code in [("bad_app.py", BAD_CODE), ("good_app.py", GOOD_CODE)]:
        findings = scan_for_secrets(code, filename)
        if findings:
            print(f"    {filename}: 警报 {len(findings)} 条!")
            for fname, lineno, label, line in findings:
                print(f"      - 第{lineno}行 [{label}] {line}")
        else:
            print(f"    {filename}: 通过 — 没有硬编码的秘密")

    # [5] 守则清单
    print("\n[5] 三重防线清单")
    print("    [预防] 秘密只放 .env/环境变量, .gitignore 里登记 .env")
    print("    [检测] 提交前用扫描器检查'秘密的模样'")
    print("    [处置] 泄露后不是删提交, 而是作废并重新签发密钥")

    print("\n小结: 别把保险柜号码(秘密)写在大门便条(代码)上。")
    print("      手册里只留一句'钥匙在储物柜里'的名牌。")


if __name__ == "__main__":
    main()
