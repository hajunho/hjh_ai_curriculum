"""
Lecture 01 / Level 10 — Docker: 環境を丸ごと再現する

Docker 無しでも Docker を学べる実習です。
1) Docker のインストール有無を診断し、2) プロジェクト仕様(辞書)から
Dockerfile のテキストを自動生成して行ごとの解説と一緒に出力し、
3) 生成した Dockerfile を outputs/ に保存し、4) ビルド時に層(layer)が
積み上がる過程をフロー図で見せます。
"""

import os
import shutil
import subprocess
from pathlib import Path

# Dockerfile を生成するプロジェクト仕様2種類 (「やってみよう」で変えてみてください)
SPECS = {
    "データ分析バッチ": {
        "python_version": "3.12",
        "packages": ["pandas", "matplotlib"],
        "entry": "analyze.py",
    },
    "Web API サーバー": {
        "python_version": "3.12",
        "packages": ["fastapi", "uvicorn"],
        "entry": "server.py",
    },
}

# 各命令の一行解説 (規格貨物のたとえ)
LINE_NOTES = {
    "FROM": "ベース貨物の選択 — Python 入りのミニ Linux から始める",
    "WORKDIR": "箱の中の作業台の位置を指定",
    "COPY": "自分のコンピュータのファイルを箱の中へ積み込む",
    "RUN": "梱包(ビルド)中に実行する組み立て作業",
    "CMD": "開封(実行)時に自動で入れるスイッチ",
}


def diagnose_docker():
    """[1] Docker インストール診断: あればバージョン確認、無ければインストール案内。"""
    print("[1] Docker インストール診断")
    docker_path = shutil.which("docker")
    if docker_path:
        print(f"    docker コマンドを発見: {docker_path}")
        result = subprocess.run(["docker", "--version"], capture_output=True, text=True, check=False)
        version = (result.stdout or result.stderr).strip()
        print(f"    バージョン: {version if version else '(確認失敗 — Docker Desktop が停止中かもしれません)'}")
        print("    -> [3]で保存する Dockerfile で 'docker build' を試せます")
        return True
    print("    docker コマンドがありません — 大丈夫、今日の実習は Docker 無しで進みます。")
    print("    インストールしたい場合: docker.com の Docker Desktop (Mac/Windows 共通)")
    print("    (会社のコンピュータならインストールポリシーを先に確認してください)")
    return False


def generate_dockerfile(spec):
    """[2] 梱包指示書の生成: 仕様の辞書 -> Dockerfile のテキスト。

    Dockerfile は結局「ルールのあるテキスト」なので、文字列の組み立てで作れます。"""
    lines = [
        f"FROM python:{spec['python_version']}-slim",
        "WORKDIR /app",
        # requirements を先にコピーする理由: 層(layer)キャッシュの活用!
        # コードだけ変わればパッケージ設置の層は再利用され、ビルドが速くなります。
        "COPY requirements.txt .",
        "RUN pip install --no-cache-dir -r requirements.txt",
        "COPY . .",
        f'CMD ["python", "{spec["entry"]}"]',
    ]
    return "\n".join(lines) + "\n"


def explain_dockerfile(name, spec):
    """生成した Dockerfile を行ごとの解説と一緒に出力します。"""
    text = generate_dockerfile(spec)
    print(f"\n    ── {name} 用の Dockerfile ──")
    print(f"    (requirements.txt: {', '.join(spec['packages'])})")
    for line in text.splitlines():
        keyword = line.split()[0] if line.strip() else ""
        note = LINE_NOTES.get(keyword, "")
        print(f"    {line:<52} # {note}" if note else f"    {line}")
    return text


def main():
    print("=" * 60)
    print("Docker 概念実習 — 梱包指示書(Dockerfile)を読み書きする")
    print("=" * 60 + "\n")

    diagnose_docker()

    # [2] 仕様別の Dockerfile 生成 + 解説
    print("\n[2] Dockerfile 生成器 — プロジェクトの仕様によって指示書が変わります")
    generated = {name: explain_dockerfile(name, spec) for name, spec in SPECS.items()}
    print("\n    比較ポイント: 2つの指示書の違いは CMD(実行ファイル)とパッケージ一覧だけ —")
    print("    構造(FROM->WORKDIR->COPY->RUN->COPY->CMD)は慣例のように同一です。")

    # [3] outputs/ フォルダへ保存 — Docker インストール後に実際にビルドを試せるように
    out_dir = Path(__file__).resolve().parent / "outputs"
    os.makedirs(out_dir, exist_ok=True)
    print("\n[3] 生成した指示書をファイルに保存")
    for name, text in generated.items():
        slug = "analysis" if "分析" in name else "webapi"
        dockerfile = out_dir / f"Dockerfile.{slug}"
        dockerfile.write_text(text, encoding="utf-8")
        print(f"    保存: {dockerfile}")
    print("    -> Docker インストール後 'docker build -f Dockerfile.analysis -t myapp .' で梱包してみてください")

    # [4] ビルドが起きたら: 層(layer)が積み上がる過程
    print("\n[4] ビルドの流れの解説 — 層(layer)が積み上がってイメージになります")
    steps = [
        ("1/6 FROM", "ベースイメージのダウンロード (最初の1回だけ、以後はキャッシュ)"),
        ("2/6 WORKDIR", "作業フォルダの層を作成"),
        ("3/6 COPY req...", "requirements.txt の層 — ファイルが変わらなければキャッシュ再利用"),
        ("4/6 RUN pip...", "パッケージ設置の層 — いちばん時間がかかるがキャッシュされれば0秒"),
        ("5/6 COPY . .", "コードの層 — コードは頻繁に変わるので最後に置く"),
        ("6/6 CMD", "実行スイッチの記録 -> イメージ完成(梱包終わり)"),
    ]
    for step, note in steps:
        print(f"    [{step:<14}] {note}")
    print("    イメージ完成後: 'docker run' = 貨物の開封・稼働(コンテナ起動)")

    print("\nまとめ: Dockerfile(指示書) -> build -> イメージ(梱包貨物) -> run -> コンテナ(稼働)。")
    print("        環境全体を貨物にすれば、どの港(コンピュータ)でも同じように動きます。")


if __name__ == "__main__":
    main()
