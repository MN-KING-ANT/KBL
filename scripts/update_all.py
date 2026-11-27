import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from collectors.kbl import update_kbl_data
from collectors.wkbl import update_wkbl_data


def main():
    print("=== KBL 데이터 업데이트 시작 ===")
    try:
        update_kbl_data(str(ROOT / "data/kbl_games.csv"), str(ROOT / "data/kbl_schedule.csv"))
    except Exception as e:
        print(f"[ERROR] KBL 업데이트 실패: {e}")

    print("\n=== WKBL 데이터 업데이트 시작 ===")
    try:
        update_wkbl_data(str(ROOT / "data/wkbl_games.csv"))
    except Exception as e:
        print(f"[ERROR] WKBL 업데이트 실패: {e}")


if __name__ == "__main__":
    main()
