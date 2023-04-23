from datetime import datetime
from typing import List

from dagster import (
    In,
    Nothing,
    OpExecutionContext,
    Out,
    ResourceDefinition,
    RetryPolicy,
    RunRequest,
    ScheduleDefinition,
    SensorEvaluationContext,
    SkipReason,
    graph,
    op,
    schedule,
    sensor,
    static_partitioned_config,
)
from workspaces.config import REDIS, S3
from workspaces.project.sensors import get_s3_keys
from workspaces.resources import mock_s3_resource, redis_resource, s3_resource
from workspaces.types import Aggregation, Stock

@op(
    config_schema={"s3_key": str}, 
    out=Out(List[Stock]), 
    required_resource_keys={"s3"}
    )
def get_s3_data(context: OpExecutionContext) -> List[Stock]:
    #pull s3_key from op config
    s3_key = context.op_config["s3_key"]

    #get s3 resource from the context
    s3 = context.resources.s3

    #read data from s3
    stock_list = []
    for record in s3.get_data(s3_key):
        stock = Stock.from_list(record)
        stock_list.append(stock)
    return stock_list


@op(
    ins={"stock_data": In(dagster_type=List[Stock])}, 
    out=Out(dagster_type=Aggregation)
    )
def process_data(context, stock_data) -> Aggregation:
    #find the max high stock value
    max_stock = max(stock_data, key=lambda stock: stock.high)
    #return the list of aggregation
    return Aggregation(date=max_stock.date, high=max_stock.high)


@op(
    ins={"aggr": In(dagster_type=Aggregation)}, 
    out=Out(),
    required_resource_keys={"redis"}
    )
def put_redis_data(context, aggr: Aggregation):
    context.resources.redis.put_data(str(aggr.date), str(aggr.high))


@op(
    ins={"aggr": In(dagster_type=Aggregation)}, 
    out=Out(),
    required_resource_keys={"s3"}
    )
def put_s3_data(context, aggr: Aggregation):
    key = f"aggr_{str(datetime.now())}.json"
    context.resources.s3.put_data(key, aggr)
    

@graph
def machine_learning_graph():
    stock_data = get_s3_data()
    max_high = process_data(stock_data)
    put_redis_data(max_high)
    put_s3_data(max_high)


local = {
    "ops": {"get_s3_data": {"config": {"s3_key": "prefix/stock_9.csv"}}},
}


docker = {
    "resources": {
        "s3": {"config": S3},
        "redis": {"config": REDIS},
    },
    "ops": {"get_s3_data": {"config": {"s3_key": "prefix/stock_9.csv"}}},
}

@static_partitioned_config(partition_keys=[str(x) for x in range(1, 11)])
def docker_config(partition_keys):
    return {
        "resources": {
            "s3": {"config": S3},
            "redis": {"config": REDIS},
            },
        "ops": {"get_s3_data": {"config": {"s3_key": f"prefix/stock_{partition_keys}.csv"}}},
    }


machine_learning_job_local = machine_learning_graph.to_job(
    name="machine_learning_job_local",
    config=local,
    resource_defs={"s3": mock_s3_resource, "redis": ResourceDefinition.mock_resource()}
)

machine_learning_job_docker = machine_learning_graph.to_job(
    name="machine_learning_job_docker",
    config=docker_config,
    resource_defs={"s3": s3_resource, "redis": redis_resource}, 
    op_retry_policy=RetryPolicy(max_retries=10, delay=1)
)



machine_learning_schedule_local = ScheduleDefinition(cron_schedule="*/15 * * * *", job=machine_learning_job_local)


@schedule(cron_schedule="0 * * * *", job=machine_learning_job_docker)
def machine_learning_schedule_docker():
    for pk in docker_config.get_partition_keys():
        yield RunRequest(run_key=pk, run_config=docker_config.get_run_config(pk))


@sensor(job = machine_learning_job_docker)
def machine_learning_sensor_docker(context):
    s3_keys = get_s3_keys(bucket = "dagster", prefix = "prefix", endpoint_url = "http://localstack:4566")
    #build sensor for 2 situations - with files or without files
    if not s3_keys:
        yield SkipReason("No new s3 files found in bucket.")
    else:
        for key in s3_keys:
            yield RunRequest(run_key = key, 
                            run_config = {
                                        "resources": {
                                            "s3": {"config": S3},
                                            "redis": {"config": REDIS},
                                            },
                                        "ops": {"get_s3_data": {"config": {"s3_key": key}}},
                                        })
