#!/usr/bin/env python3
"""
Generic grid launcher using sbatch for SLURM job submission.

➤ Submit all combos as separate sbatch jobs:
    python launch_sbatch_grid.py sweep.yaml

➤ Submit one combo only (for testing):
    python launch_sbatch_grid.py sweep.yaml --index 0

➤ Print sbatch commands without submitting:
    python launch_sbatch_grid.py sweep.yaml --print

➤ Limit concurrent jobs (good cluster citizenship):
    python launch_sbatch_grid.py sweep.yaml --max-concurrent 5
"""
import argparse, itertools, shlex, subprocess, yaml, sys, copy, time

# ------------- helper -------------------------------------------------
def dict_to_flags(d):
    """Turn a dict into ["--k", "v", ...] respecting booleans."""
    flags = []
    for k, v in d.items():
        if k == "extra_flags":          # handled elsewhere
            continue
        if isinstance(v, bool):
            if v:                       # True  -> bare flag
                flags.append(f"--{k}")
        else:                           # str, int, float
            flags.extend([f"--{k}", str(v)])
    return flags
# ----------------------------------------------------------------------

def build_commands(cfg, index=None):
    cmd_prefix = shlex.split(cfg["command"])

    # Cartesian product over runs × datasets (or runs only)
    combos = list(itertools.product(
        cfg.get("runs", [{}]),
        cfg.get("datasets", [{}])
    ))

    if index is not None:
        selected = [combos[index]]
        indices = [index]
    else:
        selected = combos
        indices = list(range(len(combos)))

    commands = []
    for combo_index, (run, dset) in zip(indices, selected):
        flags = []
        # order: base  < dataset  < run
        for section in (cfg.get("base", {}),
                        dset, run):
            flags.extend(dict_to_flags(section))
            flags.extend(section.get("extra_flags", []))

        commands.append((cmd_prefix + flags, combo_index))
    return commands


def build_env_vars(cfg, job_index):
    """Build per-job environment variables from YAML config."""
    env_cfg = cfg.get("env")
    if not env_cfg:
        return {}

    env_vars = {}
    for key, value in env_cfg.items():
        if isinstance(value, dict) and "start" in value:
            start = value["start"]
            step = value.get("step", 1)
            env_vars[key] = start + step * job_index
        elif isinstance(value, (list, tuple)):
            if not value:
                raise ValueError(f"Empty sequence provided for env var '{key}'")
            env_vars[key] = value[job_index % len(value)]
        else:
            env_vars[key] = value

    return {k: str(v) for k, v in env_vars.items()}


def build_sbatch_command(python_cmd, sbatch_config, job_index, env_vars=None):
    """Build sbatch command with the given python command."""
    sbatch_cmd = ["sbatch"]
    env_vars = {k: str(v) for k, v in (env_vars or {}).items()}
    export_value = None

    for key, value in sbatch_config.items():
        if key == "wrap_command":
            continue  # handled separately
        if key == "export":
            export_value = str(value)
            continue
        if key == "job-name":
            continue  # handled after loop
        if isinstance(value, bool):
            if value:
                sbatch_cmd.append(f"--{key}")
        else:
            sbatch_cmd.extend([f"--{key}", str(value)])

    job_name_base = sbatch_config.get("job-name", "bb-eval")
    sbatch_cmd.extend(["--job-name", f"{job_name_base}-{job_index}"])

    export_parts = []
    if export_value:
        export_parts.append(export_value.strip(','))
    elif env_vars:
        export_parts.append("ALL")

    if env_vars:
        env_str = ",".join(f"{key}={value}" for key, value in env_vars.items())
        export_parts.append(env_str)

    if export_parts:
        final_export = ",".join(part for part in export_parts if part)
        sbatch_cmd.extend(["--export", final_export])

    # Quote arguments but preserve environment variable syntax like $PORT
    quoted_args = []
    for arg in python_cmd:
        if arg.startswith('$'):
            quoted_args.append(arg)  # Don't quote environment variables
        else:
            quoted_args.append(shlex.quote(arg))
    python_cmd_str = " ".join(quoted_args)
    sbatch_cmd.extend(["--wrap", python_cmd_str])

    return sbatch_cmd

def get_running_jobs(job_name_prefix):
    """Get count of currently running/pending jobs with given name prefix."""
    try:
        result = subprocess.run(
            ["squeue", "-u", subprocess.getoutput("whoami"), "-h", "-o", "%j"],
            capture_output=True, text=True, check=True
        )
        job_names = result.stdout.strip().split('\n') if result.stdout.strip() else []
        return sum(1 for name in job_names if name.startswith(job_name_prefix))
    except subprocess.CalledProcessError:
        return 0

def wait_for_job_slots(job_name_prefix, max_concurrent):
    """Wait until there are fewer than max_concurrent jobs running."""
    while True:
        running = get_running_jobs(job_name_prefix)
        if running < max_concurrent:
            break
        print(f"⏳ Waiting... {running}/{max_concurrent} jobs running with prefix '{job_name_prefix}'")
        time.sleep(30)  # Check every 30 seconds

def main():
    p = argparse.ArgumentParser()
    p.add_argument("yaml_cfg")
    p.add_argument("--index", type=int,
                   help="Submit only combo at this flat index")
    p.add_argument("--print", action="store_true",
                   help="Print sbatch commands without submitting them")
    p.add_argument("--max-concurrent", type=int,
                   help="Maximum number of concurrent jobs to allow")
    args = p.parse_args()

    cfg = yaml.safe_load(open(args.yaml_cfg))

    # Check if sbatch config exists
    if "sbatch" not in cfg:
        sys.exit("ERROR: No 'sbatch' section found in YAML config!")

    try:
        python_cmds = build_commands(cfg, args.index)
    except IndexError:
        sys.exit(f"Index {args.index} out of range!")

    failed_submissions = []
    submitted_jobs = []
    
    # Get job name prefix for concurrent job tracking
    job_name_base = cfg["sbatch"].get("job-name", "bb-eval")
    
    total_jobs = len(python_cmds)
    for position, (python_cmd, combo_index) in enumerate(python_cmds, start=1):
        # Wait for available job slots if max_concurrent is specified
        if args.max_concurrent and not args.print:
            wait_for_job_slots(job_name_base, args.max_concurrent)

        env_vars = build_env_vars(cfg, combo_index)
        sbatch_cmd = build_sbatch_command(python_cmd, cfg["sbatch"], combo_index, env_vars)

        print(f"▶ [{position}/{total_jobs}] Submitting job:")
        print(f"Python command: {' '.join(python_cmd)}")
        print(f"Sbatch command: {' '.join(sbatch_cmd)}")

        if not args.print:
            try:
                result = subprocess.run(sbatch_cmd, check=True, capture_output=True, text=True)
                job_id = result.stdout.strip().split()[-1]  # Extract job ID from "Submitted batch job XXXXX"
                submitted_jobs.append((position, combo_index, job_id))
                print(f"✅ Job submitted: {job_id}")
            except subprocess.CalledProcessError as e:
                print(f"❌ Sbatch submission failed with exit code {e.returncode}")
                print(f"Error output: {e.stderr}")
                failed_submissions.append((position, combo_index, sbatch_cmd))
                print("Continuing to next job...\n")
                continue

    if not args.print:
        if submitted_jobs:
            print(f"\n✅ Successfully submitted {len(submitted_jobs)} job(s):")
            for position, combo_index, job_id in submitted_jobs:
                print(f"  [{position}] combo {combo_index}: Job ID {job_id}")

        if failed_submissions:
            print(f"\n⚠️  {len(failed_submissions)} job submission(s) failed:")
            for job_num, cmd in failed_submissions:
                print(f"  [{job_num}] {' '.join(cmd)}")

if __name__ == "__main__":
    main()