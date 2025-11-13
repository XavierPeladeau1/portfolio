from prefect import flow

from libs import dbt_runner


@flow
def dbt_build(
    selector: str = None,
    target: str = "dev",
    profile: str = "moka_backend",
    exclude: str = None,
):
    dbt_runner.test_models(selector, target, profile=profile, exclude=exclude)
    dbt_runner.run_models(selector, target, profile=profile, exclude=exclude)


if __name__ == "__main__":
    dbt_build("+stg_cm360_ads", target="production", profile="fizz")
