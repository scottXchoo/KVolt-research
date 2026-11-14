"""Pass a foreground workload and model-specific warmup without changing either."""
from experiment_control import environment_from_profile


def run_experiment(workload, *, profile, output_dir, warmup=None, synchronize=None):
    environment = environment_from_profile(
        profile, output_dir=output_dir, warmup=warmup, synchronize=synchronize,
    )
    result = environment.run(workload)
    return result, environment.report


# GPU callers pass their already loaded framework's synchronization callback.
# Configure session.warmup_runs in the profile when providing a warmup callback.
# Removing run_experiment and calling workload() directly detaches the module.
