import argparse
import os
import subprocess
import sys


def run_step(name, args, extra_env=None):
    print("\n" + "=" * 72)
    print(f"[{name}] {' '.join(args)}")
    print("=" * 72)
    env = os.environ.copy()
    if extra_env:
        env.update(extra_env)
    result = subprocess.run([sys.executable] + args, check=False, env=env)
    if result.returncode != 0:
        print(f"[오류] {name} 실패 (exit={result.returncode})")
        sys.exit(result.returncode)


def parse_args():
    parser = argparse.ArgumentParser(description="One-click ns-3 bridge pipeline")
    parser.add_argument("--events", default=None, help="ns-3 packet events CSV path")
    parser.add_argument("--trace", default="logs/ns3_link_trace.csv", help="bridge trace output path")
    parser.add_argument("--num-drones", type=int, default=10)
    parser.add_argument("--step-seconds", type=float, default=1.0)
    parser.add_argument("--seeds", default="42,43,44,45,46")
    parser.add_argument("--policy", choices=["trained", "random"], default="trained")
    parser.add_argument("--scenario", default="Default")
    parser.add_argument("--gif", default="logs/ns3_bridge_view.gif")
    parser.add_argument("--skip-test", action="store_true")
    parser.add_argument("--skip-viz", action="store_true")
    return parser.parse_args()


def main():
    args = parse_args()

    if args.events:
        run_step(
            "ns-3 이벤트 -> 브리지 변환",
            [
                "utils/convert_ns3_events_to_bridge_csv.py",
                "--input",
                args.events,
                "--output",
                args.trace,
                "--num-drones",
                str(args.num_drones),
                "--step-seconds",
                str(args.step_seconds),
            ],
        )

    if not os.path.exists(args.trace):
        print(f"[오류] 브리지 trace 파일이 없습니다: {args.trace}")
        print("힌트: --events를 지정하거나 --trace 경로를 확인하세요.")
        sys.exit(1)

    bridge_env = {
        "NS3_BRIDGE_ENABLED": "1",
        "NS3_BRIDGE_PROVIDER": "csv_trace",
        "NS3_BRIDGE_PATH": args.trace,
        "NS3_BRIDGE_STRICT": "0",
    }

    if not args.skip_test:
        run_step(
            "다중 시드 정책 비교",
            ["test.py", "--bridge-only", "on", "--seeds", args.seeds],
            extra_env=bridge_env,
        )

    if not args.skip_viz:
        run_step(
            "브리지 시각화",
            [
                "visualize_attack.py",
                "--policy",
                args.policy,
                "--scenario",
                args.scenario,
                "--output",
                args.gif,
                "--event-log",
                "logs/ns3_bridge_link_events.csv",
                "--no-show",
            ],
            extra_env=bridge_env,
        )

    print("\n완료: ns-3 브리지 파이프라인 실행 성공")
    print(f"- Trace: {args.trace}")
    if not args.skip_viz:
        print(f"- GIF: {args.gif}")


if __name__ == "__main__":
    main()
