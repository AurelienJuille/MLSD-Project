from kfp.dsl import (
    pipeline     # For defining the pipeline
)

from kfp import compiler
from google.cloud import aiplatform

from components.data_ingestion import data_ingestion
from components.preprocessing import preprocessing
from components.train_test import train_test_split
from components.training import training
from components.evaluation import evaluation
from components.model_deployment import deploy_model

@pipeline(
    name="lolffate pipeline",
    pipeline_root=f"gs://lolffate-data/pipeline_root"
)
def lolffate_pipeline():
    ingestion_task = data_ingestion()
    
    preprocessing_task = preprocessing(
        dataset=ingestion_task.outputs["dataset"]
    )
    
    train_test_split_task = train_test_split(
        preprocessed_dataset=preprocessing_task.outputs["preprocessed_dataset"]
    )
    
    training_task = training(
        preprocessed_dataset_train=train_test_split_task.outputs["preprocessed_dataset_train"],
    )
    
    evaluation_task = evaluation(
        model=training_task.outputs["model"],
        preprocessed_dataset_test=train_test_split_task.outputs["preprocessed_dataset_test"],
    )
    
    deploy_model_task = deploy_model(
        model=training_task.outputs["model"]
    )

def main():
    compiler.Compiler().compile(
        pipeline_func=lolffate_pipeline,
        package_path='lolffate_pipeline.json'
    )

    aiplatform.init(project="lolffate", location="europe-west1")

    pipeline_job = aiplatform.PipelineJob(
        display_name="lolffate_pipeline_job",
        template_path="lolffate_pipeline.json",
        pipeline_root=f"gs://lolffate-data/pipeline_root"
    )
    pipeline_job.run()
    
if __name__ == "__main__":
    main()
