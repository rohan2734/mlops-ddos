import mlflow

mlflow.start_run()

mlflow.log_param("param",42)
mlflow.log_metric("metrics",0.88)

active_run = mlflow.active_run()

print(f"active run id: {active_run.info.run_id}")
print(f"active run name: {active_run.info.run_name}")
print(f"active run parameters: {active_run.data.params}")
print(f"active run metrics: {active_run.data.metrics}")

mlflow.end_run()