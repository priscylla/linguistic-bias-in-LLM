"""
scripts/compute_summary.py
GPU-hours per stage, for the paper's compute statement. CPU only: reads logs
and reports, runs nothing.

    python scripts/compute_summary.py

Sources (all paths relative to the repository root):
    logs/queue/*/status, job*.log   every run_queue.sh job: wall time, command, output
    logs/rerun/*.log                jobs run by hand (no wall time logged: estimated)
    results/lens/*/*/manifest.json, results/behavioral/*/manifest.json
                                    sessions of each run (the timing check_results.py sums)
    reports/tuned_lens/*_train.json training wall time of the current lenses
    reports/tuned_lens_stale/*/*_train.json   training wall time of archived lenses
    reports/sanity/*.json, lenses/*/translators.pt   which run produced the current outputs
    reports/check_env.json          hardware and software

One job runs on one GPU (Gemma 2 27B too), so its wall time is its GPU-hours.

final       the run produced outputs that are in results/, lenses/ or reports/sanity/
            now (matched by time: the manifest session, translators.pt or sanity
            report written during the job)
discarded   superseded runs: the first Logit Lens round (measurement version 1),
            the unclipped Tuned Lenses of 2026-10-29 and their measurements, a
            manual re-measurement later redone by a queue
skipped     the job found all its outputs current and loaded no model (reported
            separately, not in the totals)
estimated   no wall time in the logs: measurement minutes from the job's own log
            plus the median model-load overhead of the same stage in the queues

Reconciliation with check_results.py: its "GPU-horas registradas" is the sum of
each manifest's gpu_hours, i.e. the measurement time of ALL sessions of that
manifest, superseded ones included (the Logit Lens manifests hold the version-1
session of 2026-10-29 and the version-2 session of 2026-10-30). The script
prints that sum split into current and superseded sessions, next to the queue
wall times; the difference is model loading and start-up.

GPU memory: the product specification (PRODUCT_MEMORY_GB); check_env.json
holds the device's total_memory in units of 10^9 bytes as the driver reports it.
"""

import json
import re
import statistics
import sys
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional

ROOT = Path(__file__).resolve().parent.parent
STAGES = ["sanity", "logit", "behavioral", "tuned_train", "tuned_eval", "tuned_remeasure"]
LABELS = {"sanity": "Sanity checks", "logit": "Logit Lens measurements",
          "behavioral": "Behavioral generation", "tuned_train": "Tuned Lens training",
          "tuned_eval": "Tuned Lens held-out evaluation (+ model loads)",
          "tuned_remeasure": "Tuned Lens re-measurement"}
SKIP_MARKS = ("modelo não carregado", "sanidade já aprovada", "já treinada e aprovada",
              "lente já treinada em")
SLACK = 120          # s de folga ao casar a janela do job com sessões e arquivos
# Memória por GPU segundo a especificação do produto (o driver reporta outro número)
PRODUCT_MEMORY_GB = {"NVIDIA H100 NVL": 94}


@dataclass
class Part:
    stage: str
    model: str
    seconds: float
    status: str               # final | discarded | skipped
    source: str
    estimated: bool = False
    note: str = ""


@dataclass
class Job:
    queue: str
    n: int
    rc: Optional[int]
    seconds: Optional[float]
    cmd: str
    log: str
    end: Optional[float]
    parts: List[Part] = field(default_factory=list)

    @property
    def start(self):
        return None if self.end is None or self.seconds is None else self.end - self.seconds


def _epoch(iso: str) -> float:
    return datetime.fromisoformat(iso).timestamp()


def _model(cmd: str) -> str:
    m = re.search(r"--model\s+(\S+?)(;|\s|$)", cmd)
    return m.group(1) if m else "?"


def _stage(cmd: str) -> str:
    if "--sanity-only" in cmd:
        return "sanity"
    if "train_tuned_lens.py" in cmd:
        return "tuned_job"
    if "run_behavioral.py" in cmd:
        return "behavioral"
    if "run_lens.py" in cmd:
        return "tuned_remeasure" if "--lens tuned" in cmd else "logit"
    return "other"


def _skipped(text: str) -> bool:
    return any(m in text for m in SKIP_MARKS) and "Carregando" not in text


def _measured_minutes(text: str) -> List[float]:
    return [float(x) for x in re.findall(r"\d+/\d+ unidades, ([\d.]+) min", text)]


# ── Fontes ──

def queue_jobs(root: Path) -> List[Job]:
    jobs = []
    for d in sorted((root / "logs" / "queue").glob("*/")):
        status = {}
        if (d / "status").exists():
            for line in (d / "status").read_text(encoding="utf-8").splitlines():
                n, rc, _gpu, sec, cmd = line.split("\t", 4)
                status[int(n)] = (int(rc), float(sec), cmd)
        cmds = [l for l in (d / "jobs").read_text(encoding="utf-8").splitlines()
                if l.strip() and not l.lstrip().startswith("#")] if (d / "jobs").exists() else []
        for log in sorted(d.glob("job*.log"), key=lambda p: int(p.stem[3:])):
            n = int(log.stem[3:])
            rc, sec, cmd = status.get(n, (None, None, cmds[n - 1] if n <= len(cmds) else "?"))
            jobs.append(Job(d.name, n, rc, sec, cmd, log.read_text(encoding="utf-8", errors="replace"),
                            log.stat().st_mtime))
    return jobs


def manual_jobs(root: Path) -> List[Job]:
    out = []
    for log in sorted((root / "logs" / "rerun").glob("*.log")):
        m = re.match(r"(.+?)_(train|tuned)$", log.stem)
        model, kind = (m.group(1), m.group(2)) if m else (log.stem, "logit")
        cmd = {"train": f"python train_tuned_lens.py --model {model}",
               "tuned": f"python run_lens.py --model {model} --lens tuned",
               "logit": f"python run_lens.py --model {model} --lens logit"}[kind]
        out.append(Job("manual (logs/rerun)", 0, None, None, cmd,
                       log.read_text(encoding="utf-8", errors="replace"), log.stat().st_mtime))
    return out


def sessions(root: Path) -> Dict[tuple, List[dict]]:
    """(stage, modelo) -> sessões com unidades, em ordem; a última é a dos resultados atuais."""
    out = {}
    for p in sorted((root / "results" / "lens").glob("*/*/manifest.json")):
        lens, model = p.parent.parent.name, p.parent.name
        stage = "logit" if lens == "logit_lens" else "tuned_remeasure"
        out[(stage, model)] = [s for s in json.loads(p.read_text())["sessions"] if s.get("units_run")]
    for p in sorted((root / "results" / "behavioral").glob("*/manifest.json")):
        out[("behavioral", p.parent.name)] = [s for s in json.loads(p.read_text())["sessions"]
                                              if s.get("units_run")]
    return out


def train_reports(root: Path) -> Dict[str, List[tuple]]:
    """modelo -> [(wall_time_s, 'final'|'archived', arquivo)]."""
    out = {}
    for p in sorted((root / "reports" / "tuned_lens").glob("*_train.json")):
        out.setdefault(p.name[:-len("_train.json")], []).append(
            (json.loads(p.read_text())["wall_time_s"], "final", str(p.relative_to(root))))
    for p in sorted((root / "reports").glob("tuned_lens_stale/*/*_train.json")):
        out.setdefault(p.name[:-len("_train.json")], []).append(
            (json.loads(p.read_text())["wall_time_s"], "archived", str(p.relative_to(root))))
    return out


# ── Classificação ──

def _in(job: Job, t: float) -> bool:
    return job.start is not None and job.start - SLACK <= t <= job.end + SLACK


def _session_status(job: Job, sess: List[dict]) -> Optional[tuple]:
    """(status, sessão) da sessão do manifest feita durante o job."""
    for i, s in enumerate(sess):
        if _in(job, _epoch(s["started"])):
            return ("final" if i == len(sess) - 1 else "discarded"), s
    return None


def classify(jobs: List[Job], root: Path) -> None:
    sess, trains = sessions(root), train_reports(root)
    overhead = {}           # stage -> [job s − sessão s] (carregar modelo etc.), para estimar
    for job in jobs:
        if job.seconds is None:
            continue
        st = _stage(job.cmd)
        if st in ("logit", "behavioral", "tuned_remeasure"):
            hit = _session_status(job, sess.get((st, _model(job.cmd)), []))
            if hit:
                overhead.setdefault(st, []).append(job.seconds - hit[1]["wall_time_s"])
    med = {k: statistics.median(v) for k, v in overhead.items()}

    for job in jobs:
        st, model = _stage(job.cmd), _model(job.cmd)
        if job.seconds is None and job.queue != "manual (logs/rerun)":
            job.parts.append(Part(st if st != "tuned_job" else "tuned_train", model, 0.0, "hung",
                                  f"{job.queue} job {job.n}",
                                  note="no status line and " + ("an empty log" if not job.log.strip()
                                                                else "no completion") +
                                       ": duration not logged, no model loaded"))
            continue
        src = f"{job.queue} job {job.n}" if job.n else job.queue
        if st == "sanity":
            rep = root / "reports" / "sanity" / f"{model}.json"
            if _skipped(job.log):
                status = "skipped"
            else:
                status = "final" if rep.exists() and _in(job, rep.stat().st_mtime) else "discarded"
            job.parts.append(Part("sanity", model, job.seconds, status, src))
        elif st in ("logit", "behavioral", "tuned_remeasure"):
            if job.seconds is None:                                    # manual: estimate
                mins = _measured_minutes(job.log)
                if not mins:
                    job.parts.append(Part(st, model, 0.0, "skipped", src, note="no measurement in log"))
                    continue
                hit = _session_status_by_end(job, sess.get((st, model), []))
                status = hit[0] if hit else "discarded"
                load = med.get(st, med.get("logit", 0.0))
                secs = sum(mins) * 60 + load
                job.parts.append(Part(st, model, secs, status, src, estimated=True,
                                      note=f"{sum(mins):.1f} min measured + {load:.0f} s median load"))
                continue
            if _skipped(job.log):
                job.parts.append(Part(st, model, job.seconds, "skipped", src))
                continue
            hit = _session_status(job, sess.get((st, model), []))
            job.parts.append(Part(st, model, job.seconds, hit[0] if hit else "discarded", src,
                                  note="" if hit else "outputs not in current results (superseded)"))
        elif st == "tuned_job":
            if job.seconds is None:                                    # manual train log: skip
                job.parts.append(Part("tuned_train", model, 0.0, "skipped", src))
                continue
            lens = root / "lenses" / model / "translators.pt"
            final = lens.exists() and _in(job, lens.stat().st_mtime)
            status = "final" if final else "discarded"
            if _skipped(job.log) and "Carregando" not in job.log:
                job.parts.append(Part("tuned_train", model, job.seconds, "skipped", src))
                continue
            cands = [t for t in trains.get(model, []) if t[1] == ("final" if final else "archived")]
            train_s = cands[0][0] if cands else None
            # remedição dentro do job: sessão do manifest atual, ou os minutos do log
            hit = _session_status(job, sess.get(("tuned_remeasure", model), []))
            mins = _measured_minutes(job.log)
            load = med.get("tuned_remeasure", med.get("logit", 0.0))   # carregar o modelo de novo
            if hit:
                rem_s, rem_status = hit[1]["wall_time_s"] + load, hit[0]
            elif mins:
                rem_s, rem_status = sum(mins) * 60 + load, "discarded"
            else:
                rem_s, rem_status = 0.0, status
            if train_s is None:
                job.parts.append(Part("tuned_train", model, job.seconds - rem_s, status, src, estimated=True,
                                      note="no *_train.json: training and evaluation not separated"))
            else:
                job.parts.append(Part("tuned_train", model, train_s, status, src,
                                      note=cands[0][2]))
                job.parts.append(Part("tuned_eval", model, max(job.seconds - train_s - rem_s, 0.0), status, src,
                                      note="job wall − training − re-measurement"))
            if rem_s:
                job.parts.append(Part("tuned_remeasure", model, rem_s, rem_status, src,
                                      estimated=not hit, note="inside the training job"))


def _session_status_by_end(job: Job, sess: List[dict]) -> Optional[tuple]:
    """Para jobs manuais (sem duração): a sessão que termina perto do fim do log."""
    for i, s in enumerate(sess):
        if abs(_epoch(s["finished"]) - job.end) <= 3600:
            return ("final" if i == len(sess) - 1 else "discarded"), s
    return None


# ── Relatório ──

def hardware(root: Path) -> Dict:
    out = {"gpus": [], "torch": None, "cuda": None, "driver": None, "transformers": None, "python": None}
    p = root / "reports" / "check_env.json"
    if p.exists():
        d = json.loads(p.read_text())
        out["python"] = d.get("python")
        for c in d.get("checks", []):
            if re.fullmatch(r"gpu \d+", c.get("name", "")):
                m = re.match(r"(.+?), ([\d.]+) GB", c["detail"])
                if m:
                    out["gpus"].append((m.group(1), float(m.group(2))))
            if c.get("name") == "torch":
                m = re.match(r"(\S+), CUDA build ([\d.]+), NVIDIA driver (\S+)", c["detail"])
                if m:
                    out["torch"], out["cuda"], out["driver"] = m.groups()
    for p in sorted((root / "results").glob("**/manifest.json")):
        v = json.loads(p.read_text()).get("versions") or {}
        if v.get("transformers"):
            out["transformers"] = v["transformers"]
            break
    return out


def reconcile(root: Path, parts: List[Part]) -> List[Dict]:
    """
    Por etapa medida por run_lens / run_behavioral: o que check_results.py soma
    (todas as sessões dos manifests), separado em sessões atuais e substituídas,
    ao lado do tempo de parede dos jobs final / discarded.
    """
    sess = sessions(root)
    rows = []
    for stage in ("logit", "tuned_remeasure", "behavioral"):
        cur = sum(ss[-1]["wall_time_s"] for (st, _), ss in sess.items() if st == stage and ss)
        allw = sum(s["wall_time_s"] for (st, _), ss in sess.items() if st == stage for s in ss)
        jf = sum(p.seconds for p in parts if p.stage == stage and p.status == "final")
        jd = sum(p.seconds for p in parts if p.stage == stage and p.status == "discarded")
        rows.append({"stage": stage, "check_results_h": allw / 3600, "current_sessions_h": cur / 3600,
                     "superseded_sessions_h": (allw - cur) / 3600, "jobs_final_h": jf / 3600,
                     "jobs_discarded_h": jd / 3600})
    return rows


def summarize(root: Path = ROOT) -> Dict:
    jobs = queue_jobs(root) + manual_jobs(root)
    classify(jobs, root)
    parts = [p for j in jobs for p in j.parts]
    return {"parts": parts, "hardware": hardware(root), "reconcile": reconcile(root, parts)}


def _h(s):
    return s / 3600


def report(summary: Dict) -> str:
    parts = summary["parts"]
    L = ["GPU-hours per stage (one job = one GPU; wall time = GPU-hours)", ""]
    head = f"{'Stage':48} {'Model':13} {'Final':>8} {'Discarded':>10}  Notes"
    L += [head, "-" * len(head)]
    tot = {"final": 0.0, "discarded": 0.0, "skipped": 0.0}
    for stage in STAGES:
        ps = [p for p in parts if p.stage == stage and p.status in ("final", "discarded")]
        by_model = stage in ("behavioral", "tuned_train", "tuned_eval")
        groups = sorted({p.model for p in ps}) if by_model else ["all"]
        for g in groups:
            sel = [p for p in ps if by_model is False or p.model == g]
            f = sum(p.seconds for p in sel if p.status == "final")
            d = sum(p.seconds for p in sel if p.status == "discarded")
            est = [p for p in sel if p.estimated]
            note = (f"estimated: {', '.join(sorted({p.model + ' (' + p.note + ')' for p in est}))}" if est else "")
            label = LABELS[stage] if g == groups[0] else ""
            L.append(f"{label:48} {g:13} {_h(f):8.2f} {_h(d):10.2f}  {note}")
            tot["final"] += f
            tot["discarded"] += d
        if not ps:
            L.append(f"{LABELS[stage]:48} {'—':13} {'—':>8} {'—':>10}  no timing in the logs")
    L.append("-" * len(head))
    L.append(f"{'Total':48} {'':13} {_h(tot['final']):8.2f} {_h(tot['discarded']):10.2f}")
    L.append(f"{'All (final + discarded)':48} {'':13} {_h(tot['final'] + tot['discarded']):8.2f}")
    sk = [p for p in parts if p.status == "skipped" and p.seconds]
    L += ["", f"Skipped jobs (all outputs already current, no model loaded): {len(sk)} jobs, "
              f"{_h(sum(p.seconds for p in sk)):.2f} h of wall time (start-up only; not in the totals)"]
    hung = [p for p in parts if p.status == "hung"]
    for p in hung:
        L.append(f"Hung job: {p.source}, {p.model} ({LABELS.get(p.stage, p.stage)}): {p.note}; not in the totals")
    est = [p for p in parts if p.estimated and p.status in ("final", "discarded")]
    if est:
        L.append(f"Estimated parts: {len(est)} ({_h(sum(p.seconds for p in est)):.2f} h), marked above")
    L += ["", "Reconciliation with check_results.py (manifest measurement time, GPU-h)",
          f"{'Stage':28} {'check_results':>13} {'= current':>10} {'+ superseded':>13} "
          f"{'jobs final':>11} {'jobs discarded':>15}"]
    for r in summary["reconcile"]:
        L.append(f"{LABELS[r['stage']][:28]:28} {r['check_results_h']:13.2f} {r['current_sessions_h']:10.2f} "
                 f"{r['superseded_sessions_h']:13.2f} {r['jobs_final_h']:11.2f} {r['jobs_discarded_h']:15.2f}")
    L.append("check_results sums every session of each manifest, superseded ones included; jobs final − "
             "current sessions and jobs discarded − superseded sessions are model loading and start-up.")
    hw = summary["hardware"]
    gpus = hw["gpus"]
    names = sorted({n for n, _ in gpus})
    if len(names) == 1:
        spec = PRODUCT_MEMORY_GB.get(names[0])
        mem = (f", {spec} GB each (product specification; the driver reports "
               f"{gpus[0][1]:.1f} × 10^9 bytes)" if spec else f", {gpus[0][1]:.1f} × 10^9 bytes each")
        hw_line = f"{len(gpus)} × {names[0]}{mem}"
    else:
        hw_line = "; ".join(f"{n} ({g:.1f} × 10^9 bytes)" for n, g in gpus) or "—"
    L += ["", "Hardware: " + hw_line,
          f"Software: torch {hw['torch']}, CUDA {hw['cuda']} (driver {hw['driver']}), "
          f"transformers {hw['transformers']}, Python {hw['python']}"]
    return "\n".join(L)


def main(argv=None):
    print(report(summarize(ROOT)))
    return 0


if __name__ == "__main__":
    sys.exit(main())
