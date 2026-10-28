"""
monitor.py
Monitor de progresso dos experimentos em tempo real.

Executar em terminal separado enquanto run_lens.py roda:
    python monitor.py

Atualiza a cada 30 segundos.
"""

import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from config.experiment_config import ExperimentConfig
from logit_lens.store import ChunkStore, lens_dir, read_manifest
from run_lens import lens_units


def count_completed_runs(results_dir: str = None) -> dict:
    """
    Unidades feitas (fato × idioma e linha de base × idioma) por modelo.
    O total esperado vem do config.
    """
    config      = ExperimentConfig()
    results_dir = results_dir or config.results_dir
    summary     = {}

    for model_key in config.models:
        out   = lens_dir(results_dir, model_key)
        man   = read_manifest(out)
        total = len(lens_units(config, tuple(man.get("formats", ["completion", "qa"]))))
        done  = len(ChunkStore(out).done_units()) if out.exists() else 0
        status = "pending" if done == 0 else ("done" if done >= total else "running")
        summary[model_key] = {"status": status, "completed": done, "total": total,
                              "pct": done / total * 100 if total else 0.0}
    return summary


def monitor_gpu() -> list:
    """Lê uso de GPU via nvidia-smi."""
    result = os.popen(
        "nvidia-smi --query-gpu=name,memory.used,memory.total,utilization.gpu "
        "--format=csv,noheader,nounits 2>/dev/null"
    ).read()

    gpus = []
    for line in result.strip().split("\n"):
        if line:
            parts = [p.strip() for p in line.split(",")]
            if len(parts) == 4:
                try:
                    gpus.append({
                        "name":      parts[0],
                        "mem_used":  int(parts[1]),
                        "mem_total": int(parts[2]),
                        "util":      int(parts[3])
                    })
                except ValueError:
                    pass
    return gpus


def display_status():
    """Exibe status formatado no terminal."""
    os.system("clear")

    print("=" * 60)
    print("MONITOR DE EXPERIMENTO")
    print(f"Atualizado: {time.strftime('%H:%M:%S')}")
    print("=" * 60)

    # ── Progresso dos runs ──
    print("\n📊 PROGRESSO DOS RUNS:")
    summary   = count_completed_runs()
    total_done = 0
    total_all  = 0

    for model_key, info in summary.items():
        bar_len   = 28
        filled    = int(bar_len * info["pct"] / 100)
        bar       = "█" * filled + "░" * (bar_len - filled)

        status_icon = {
            "pending": "⏳",
            "running": "🔄",
            "writing": "✍️ ",
            "done":    "✅"
        }.get(info["status"], "?")

        print(
            f"  {status_icon} {model_key:8s} [{bar}] "
            f"{info['completed']:2}/{info['total']} "
            f"({info['pct']:5.1f}%)"
        )

        if isinstance(info["completed"], int):
            total_done += info["completed"]
        total_all += info["total"]

    pct_total = total_done / total_all * 100 if total_all > 0 else 0
    print(f"\n  TOTAL: {total_done}/{total_all} unidades ({pct_total:.1f}%)")

    # ── Status das GPUs ──
    print("\n🖥️  STATUS DAS GPUs:")
    gpus = monitor_gpu()

    if not gpus:
        print("  (nvidia-smi não disponível)")
    else:
        for i, gpu in enumerate(gpus):
            mem_pct = gpu["mem_used"] / gpu["mem_total"] * 100
            bar_len = 18
            filled  = int(bar_len * mem_pct / 100)
            bar     = "█" * filled + "░" * (bar_len - filled)

            print(
                f"  GPU {i}: {gpu['name'][:22]:22s} "
                f"[{bar}] {gpu['mem_used']:5d}/{gpu['mem_total']:5d} MB "
                f"({mem_pct:.0f}%)  "
                f"Util: {gpu['util']:3d}%"
            )

    print("\n  [Ctrl+C para sair do monitor]")


if __name__ == "__main__":
    print("Iniciando monitor... (atualiza a cada 30s)")
    try:
        while True:
            display_status()
            time.sleep(30)
    except KeyboardInterrupt:
        print("\n\nMonitor encerrado.")
