import subprocess
import sys


def run_step(name, args):
    print("\n" + "=" * 70)
    print(f"  [{name}] 실행 중: {' '.join(args)}")
    print("=" * 70)

    result = subprocess.run([sys.executable] + args, check=False)
    if result.returncode != 0:
        print(f"[오류] {name} 단계가 실패했습니다. 종료 코드: {result.returncode}")
        sys.exit(result.returncode)
    print(f"[완료] {name} 단계 성공")


def main():
    steps = [
        ("학습", ["train.py"]),
        ("평가", ["eval.py"]),
        ("시각화", ["utils/generate_all_plots.py"]),
    ]

    for name, args in steps:
        run_step(name, args)

    print("\n모든 단계가 완료되었습니다. logs/ 디렉터리와 models/weights/를 확인하세요.")


if __name__ == "__main__":
    main()
