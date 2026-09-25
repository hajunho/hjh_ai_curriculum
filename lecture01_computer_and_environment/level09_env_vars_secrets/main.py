"""
Lecture 01 / Level 09 — 환경변수와 비밀키 안전하게 다루기

비밀을 코드 밖에서 다루는 세 가지 도구를 실습합니다.
1) os.environ 으로 환경변수 읽기, 2) 비밀값 마스킹 출력,
3) .env 파일 파서(dotenv 원리), 4) 하드코딩된 비밀을 잡아내는 미니 스캐너.
모든 키 값은 실습용 가짜이며 인터넷에 연결하지 않습니다.
"""

import os
import re

# 하드코딩 비밀 탐지 규칙: (설명, 정규표현식) — 실제 보안 도구의 축소판
SECRET_PATTERNS = [
    ("API 키 모양 (sk-...)", re.compile(r"sk-[A-Za-z0-9]{8,}")),
    ("password= 하드코딩", re.compile(r"password\s*=\s*['\"][^'\"]+['\"]", re.IGNORECASE)),
    ("secret/token 하드코딩", re.compile(r"(secret|token)\s*=\s*['\"][^'\"]{6,}['\"]", re.IGNORECASE)),
]

# [4]에서 검사할 예제 코드 — 나쁜 예(비밀이 코드 안에)와 좋은 예(이름만 참조)
BAD_CODE = '''
# bad_app.py — 이렇게 쓰면 안 됩니다!
api_key = "sk-demo1234567890abcdef"
password = "corp!2024"
db = connect("10.0.0.7", password=password)
'''
GOOD_CODE = '''
# good_app.py — 비밀은 이름으로만 참조합니다
import os
api_key = os.environ["LLM_API_KEY"]      # 값은 환경변수(사물함)에
password = os.environ.get("DB_PASSWORD")  # 코드에는 이름표만
'''


def mask(value, show=4):
    """비밀값을 'sk-d****5678' 형태로 가립니다. 확인은 하되 노출은 막기."""
    if value is None:
        return "(설정 안 됨)"
    if len(value) <= show * 2:
        return "*" * len(value)
    return value[:show] + "*" * 4 + value[-show:]


def parse_dotenv(text):
    """[3] .env 파서: '이름=값' 형식 텍스트를 딕셔너리로. dotenv 도구의 원리."""
    result = {}
    for line in text.splitlines():
        line = line.strip()
        if not line or line.startswith("#"):   # 빈 줄과 주석은 건너뜀
            continue
        name, _, value = line.partition("=")
        result[name.strip()] = value.strip().strip('"').strip("'")
    return result


def scan_for_secrets(code, filename):
    """[4] 미니 스캐너: 코드에서 '비밀의 모양'을 한 문자열을 찾아냅니다."""
    findings = []
    for lineno, line in enumerate(code.splitlines(), start=1):
        for label, pattern in SECRET_PATTERNS:
            if pattern.search(line):
                findings.append((filename, lineno, label, line.strip()))
    return findings


def main():
    print("=" * 60)
    print("환경변수와 비밀키 — 비밀은 코드 밖에, 코드에는 이름만")
    print("=" * 60)

    # [1] 환경변수 읽기: 실습용 변수를 이 프로세스에 설정하고 읽어 봅니다
    os.environ["DEMO_LLM_API_KEY"] = "sk-demo1122334455667788"  # 실습용 가짜 값
    print("\n[1] 환경변수 읽기 — 운영체제가 건네주는 '이름=값' 메모지")
    print(f"    os.environ['DEMO_LLM_API_KEY'] -> (읽기 성공, 값은 아래서 마스킹)")
    print(f"    .get() 안전 읽기: 없는 변수 -> {os.environ.get('NO_SUCH_VAR')}")
    print(f"    이미 쓰고 있던 환경변수 PATH 의 길이: {len(os.environ.get('PATH', ''))}글자")
    practice = os.environ.get("PRACTICE_KEY")
    print(f"    PRACTICE_KEY: {mask(practice)}  (직접 해보기 1번에서 export 해 보세요)")

    # [2] 마스킹 출력: '설정 여부'만 보여주고 값은 가립니다
    print("\n[2] 마스킹 출력 — 로그·화면에 비밀을 통째로 찍지 않기")
    secret = os.environ["DEMO_LLM_API_KEY"]
    print(f"    원래 값 그대로 출력  : (절대 금지!)")
    print(f"    마스킹해서 출력     : {mask(secret)}")

    # [3] .env 파서: 파일 내용을 환경변수로 올리는 원리
    print("\n[3] .env 패턴 — 프로젝트 폴더의 비밀 보관 파일 (Git 에는 절대 금지)")
    dotenv_text = '# 실습용 .env 내용\nDB_PASSWORD="s3cret!pw"\nSLACK_TOKEN=xoxb-demo-9988\n'
    loaded = parse_dotenv(dotenv_text)
    for name, value in loaded.items():
        os.environ[name] = value           # 환경변수로 승격 (dotenv 가 하는 일)
        print(f"    {name:<14} = {mask(value)}  -> os.environ 에 올림")
    print("    -> 코드는 이제 os.environ['DB_PASSWORD'] 로 이름만 부르면 됩니다")

    # [4] 하드코딩 스캐너: 나쁜 코드에서만 경보가 울려야 합니다
    print("\n[4] 하드코딩 탐지 미니 스캐너 — 커밋 전 자동 검사의 원리")
    for filename, code in [("bad_app.py", BAD_CODE), ("good_app.py", GOOD_CODE)]:
        findings = scan_for_secrets(code, filename)
        if findings:
            print(f"    {filename}: 경보 {len(findings)}건!")
            for fname, lineno, label, line in findings:
                print(f"      - {lineno}행 [{label}] {line}")
        else:
            print(f"    {filename}: 통과 — 하드코딩된 비밀 없음")

    # [5] 수칙 체크리스트
    print("\n[5] 3중 방어선 체크리스트")
    print("    [예방] 비밀은 .env/환경변수로만, .gitignore 에 .env 등록")
    print("    [탐지] 커밋 전 스캐너로 '비밀의 모양' 검사")
    print("    [대응] 유출됐다면 커밋 삭제가 아니라 키 폐기·재발급")

    print("\n정리: 현관문 메모지(코드)에 금고 번호(비밀)를 적지 마세요.")
    print("      매뉴얼에는 '열쇠는 사물함에'라는 이름표만 남기는 것입니다.")


if __name__ == "__main__":
    main()
