"""
データプロジェクトの 5 段階 (問題定義→データ→モデル→デプロイ→モニタリング) を
関門 (gate) チェックリストで通過させるシミュレーション。
架空の「サブスク解約防止」プロジェクトを例に、
各段階の成果物が空欄のままだと関門で STOP 判定が出ることを観察します。
"""

import sys
import pathlib

sys.path.append(str(pathlib.Path(__file__).resolve().parents[2] / "common"))
import hjh_data


def gate(stage_name: str, checklist: dict) -> bool:
    """チェックリストの値が 1 つでも空なら関門の通過は失敗。"""
    print(f"\n  [{stage_name}] 関門チェック")
    ok = True
    for item, value in checklist.items():
        filled = value not in ("", None, [])
        mark = "OK " if filled else "欠落"
        shown = value if filled else "(空欄)"
        print(f"    - {mark} | {item}: {shown}")
        if not filled:
            ok = False
    print(f"    => 判定: {'PASS - 次の段階へ進む' if ok else 'STOP - 埋めるまで進行禁止'}")
    return ok


def main() -> None:
    print("=" * 62)
    print(" データプロジェクト・ライフサイクルのシミュレーション: 「サブスク解約防止」プロジェクト")
    print("=" * 62)
    results = {}

    # ------------------------------------------------------------------
    print("\n[1] 問題定義 — 何を、なぜ、成功基準は?")
    problem_spec = {
        "予測対象(ラベル定義)": "翌月の決済更新をしなかった顧客 = 解約(1)",
        "予測結果の活用方法": "毎週月曜にリスク上位200名へ相談/クーポンを提供",
        "目標指標(数字)": "6か月以内に月次解約率 18% -> 15%",
        "ベースライン(現在のやり方)": "全顧客から無作為に100名へクーポンを送付",
    }
    results["1.問題定義"] = gate("問題定義", problem_spec)

    # ------------------------------------------------------------------
    print("\n[2] データ — その問いに答えられるデータはあるか? (実際に監査を実施)")
    rows = hjh_data.churn_table(n=2000, seed=7)
    n_total = len(rows)
    n_missing = sum(1 for r in rows if any(v is None for v in r.values()))
    n_churn = sum(r["churned"] for r in rows)
    churn_rate = n_churn / n_total
    min_minority = 200  # 少数クラス(解約)の最小サンプル数の関門基準

    print(f"    行数: {n_total} / 欠損のある行: {n_missing}")
    print(f"    解約顧客: {n_churn}名 (解約率 {churn_rate:.1%})")
    data_audit = {
        "サンプルサイズの確認": f"{n_total}行",
        "欠損チェック": f"{n_missing}行 (許容範囲)",
        "少数クラスのサンプル": f"{n_churn}名" if n_churn >= min_minority else "",
        "ラベル定義の一致確認": "決済ログ基準、CSチームと合意済み",
    }
    results["2.データ"] = gate("データ", data_audit)

    # ------------------------------------------------------------------
    print("\n[3] モデル — 超えるべきベースライン (baseline) をまず計算")
    # モデルなしで「多数クラス(非解約)」とだけ答えるベースラインの精度
    majority_acc = 1 - churn_rate
    print(f"    多数決ベースラインの精度: {majority_acc:.1%}  <- 「誰も解約しない」と言うだけでこの数字が出ます")
    print("    => この後作るモデルは「精度」ではなく、解約者を実際に見つけ出す")
    print("       再現率/適合率でこのベースラインに勝たなければなりません (level01, level07 に続く)。")
    model_report = {
        "ベースライン性能の記録": f"多数決の精度 {majority_acc:.1%}",
        "検証方法の合意": "交差検証 5-fold (level06 で学習)",
        "モデル性能の報告": "(以降のレベルで作成予定)",  # デモなので埋まっている扱い
    }
    results["3.モデル"] = gate("モデル", model_report)

    # ------------------------------------------------------------------
    print("\n[4] デプロイ — 現場が実際に使えるか?")
    deploy_plan = {
        "結果の伝達チャネル": "毎週月曜に CRM システムへリスク顧客リストをアップロード",
        "受け取る人と業務手順": "",  # わざと空欄: STOP 判定を観察するため
        "障害時の対応(モデル停止時)": "",
    }
    results["4.デプロイ"] = gate("デプロイ", deploy_plan)

    # ------------------------------------------------------------------
    print("\n[5] モニタリング — 性能は維持されているか?")
    monitor_plan = {
        "性能の追跡指標": "週次の再現率/適合率、キャンペーン後の実際の解約率",
        "データ分布シフトの監視": "",  # わざと空欄
        "再学習の基準": "再現率が2週連続で5%pt以上低下したら再学習",
    }
    results["5.モニタリング"] = gate("モニタリング", monitor_plan)

    # ------------------------------------------------------------------
    print("\n" + "=" * 62)
    print("[6] 最終サマリー — このプロジェクトは今どこで止まるべきか")
    print("=" * 62)
    first_stop = None
    for stage, ok in results.items():
        print(f"    {stage:<10} : {'PASS' if ok else 'STOP'}")
        if not ok and first_stop is None:
            first_stop = stage
    if first_stop:
        print(f"\n    => 最初の STOP 地点: {first_stop}")
        print("       モデルの性能をさらに上げるより、この関門の空欄を埋めることが先決です。")
    print("\n    教訓: 失敗するプロジェクトはコードではなく、「空欄を抱えたまま前進」して崩れます。")


if __name__ == "__main__":
    main()
