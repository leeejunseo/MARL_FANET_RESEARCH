import argparse
import csv
import os


def parse_args():
    parser = argparse.ArgumentParser(description="Generate a mock ns-3 style link trace CSV.")
    parser.add_argument("--output", default="logs/ns3_link_trace.csv")
    parser.add_argument("--num-drones", type=int, default=3)
    parser.add_argument("--steps", type=int, default=60)
    return parser.parse_args()


def link_state(step, i, j):
    # Deterministic pattern for quick bridge tests.
    if i == j:
        return True, 0.0, 1.0, 0.0

    pair = tuple(sorted((i, j)))
    if pair == (0, 1):
        connected = not (15 <= step <= 25)
    elif pair == (1, 2):
        connected = not (35 <= step <= 45)
    else:
        connected = True

    if connected:
        delay_ms = 20.0 + 1.2 * step
        delivery = 0.92
        hop = 1.0
    else:
        delay_ms = 300.0
        delivery = 0.0
        hop = 99.0

    return connected, delay_ms, delivery, hop


def main():
    args = parse_args()
    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)

    with open(args.output, "w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["step", "i", "j", "connected", "delay_ms", "delivery", "hop"])
        for step in range(1, args.steps + 1):
            for i in range(args.num_drones):
                for j in range(args.num_drones):
                    connected, delay_ms, delivery, hop = link_state(step, i, j)
                    writer.writerow([step, i, j, int(connected), f"{delay_ms:.2f}", f"{delivery:.4f}", f"{hop:.1f}"])

    print(f"Mock ns-3 trace generated: {args.output}")


if __name__ == "__main__":
    main()
