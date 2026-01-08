import subprocess

from mrunner.helpers.client_helper import get_configuration

if __name__ == "__main__":
    cfg = get_configuration(print_diagnostics=True, with_neptune=False)

    entry_point = cfg.pop("entry_point", "train")
    cfg.pop("experiment_id")

    key_pairs = []
    for key, value in cfg.items():
        if isinstance(value, bool):
            key_list = key.split(".")
            key_list[-1] = key_list[-1] if value else f"no-{key_list[-1]}"
            key = ".".join(key_list)
            key_pairs.append(f"--{key}")
        elif isinstance(value, list):
            key_pairs.append(f"--{key}")
            for item in value:
                key_pairs.append(str(item))
        else:
            key_pairs.extend([f"--{key}", str(value)])

    cmd = ["python", "-m", entry_point] + key_pairs
    print("Running command:", cmd)
    subprocess.run(cmd)