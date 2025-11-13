import json
import os
import subprocess
from typing import List
from prefect import task
from prefect.blocks.system import Secret
from prefect_dbt.cli.commands import DbtCoreOperation


def _run_command(command: List[str], cwd: str):
    """Run a command in a subprocess, raising an exception if it fails."""

    # If snowflake_password is not already in env, load it from Prefect Secrets
    env_vars = os.environ.copy()
    if env_vars.get("SNOWFLAKE_PASSWORD") is None:
        env_vars["SNOWFLAKE_PASSWORD"] = Secret.load("snowflake-password").get()

    process = subprocess.run(command, capture_output=True, cwd=cwd, env=env_vars)

    print(process.stdout.decode("utf-8"))
    if process.stderr:
        print(process.stderr.decode("utf-8"))

    if process.returncode != 0:
        raise Exception(f"Command {command} failed with {process.returncode}")


@task(log_prints=True, task_run_name="DBT build : {node_name}")
def run_models(node_name, target, profile="moka_backend", vars={}, exclude=None):
    args = [
        "dbt",
        "run",
        "--select",
        node_name,
        "--vars",
        json.dumps(vars),
        "--target",
        target,
    ]

    if exclude is not None:
        args.extend(["--exclude", exclude])

    result = DbtCoreOperation(
        commands=[" ".join(args)],
        profiles_dir=f"./{profile}",
        project_dir=f"./{profile}",
    ).run()

    return result


def test_models(node_name, target, profile, vars={}, exclude=None):
    args = [
        "dbt",
        "test",
        "--select",
        node_name,
        "--vars",
        json.dumps(vars),
        "--target",
        target,
    ]

    if exclude is not None:
        args.extend(["--exclude", exclude])

    result = DbtCoreOperation(
        commands=[" ".join(args)],
        profiles_dir=f"./{profile}",
        project_dir=f"./{profile}",
    ).run()

    return result


def run_operation(operation, target, profile, args={}):
    args = [
        "dbt",
        "run-operation",
        operation,
        "--args",
        json.dumps(args),
        "--profiles-dir",
        ".",
        "--target",
        target,
    ]

    _run_command(args, cwd=f"./{profile}")
