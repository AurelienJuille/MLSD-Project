from kfp.dsl import (
    Input,       # For component inputs
    Model,       # For handling ML models
    Output,      # For component outputs
    component,   # For creating pipeline components
)

@component(
    base_image=f"europe-west1-docker.pkg.dev/lolffate/lolffate-pipeline/training:latest",
    install_kfp_package=False,
    output_component_file="model_deployment.yaml",
)

def deploy_model(
        model: Input[Model],
        vertex_model: Output[Model],
        vertex_endpoint: Output[Model]
):
    """
    Deploys the trained model to Vertex AI and creates an endpoint.
    
    Args:
        model: Input trained model
        vertex_model: Output artifact for the deployed model
        vertex_endpoint: Output artifact for the Vertex AI endpoint
    """
    from google.cloud import aiplatform as vertex_ai
    from pathlib import Path
    import logging
    
    # Checks existing Vertex AI Enpoint or creates Endpoint if it is not exist.
    def create_endpoint ():
        endpoints = vertex_ai.Endpoint.list(
            filter='display_name=lolffate-endpoint',
            order_by='create_time desc',
            project='lolffate',
            location='europe-west1',
        )
        
        if len(endpoints) > 0:
            endpoint = endpoints[0] # most recently created
        else:
            endpoint = vertex_ai.Endpoint.create(
                display_name='lolffate-endpoint',
                project='lolffate',
                location='europe-west1'
            )
            
        return endpoint

    endpoint = create_endpoint()
    logging.info(f"Using endpoint: {endpoint.resource_name}")
    
    # Uploads trained model version to Vertex AI Model Registry
    def upload_model ():   
        listed_model = vertex_ai.Model.list(
            filter='display_name=lolffate',
            project='lolffate',
            location='europe-west1',
        )
        
        logging.info(f"Uploading model from artifact URI: {model.path}")
        
        if len(listed_model) > 0:
            model_version = listed_model[0] # most recently created
            model_upload = vertex_ai.Model.upload(
                display_name='lolffate',
                parent_model=model_version.resource_name, # add link to the previously model version
                artifact_uri=model.path,
                serving_container_image_uri='europe-docker.pkg.dev/vertex-ai/prediction/sklearn-cpu.1-5:latest',
                location='europe-west1'
            )
        else:
            model_upload = vertex_ai.Model.upload(
                display_name='lolffate',
                artifact_uri=model.path,
                serving_container_image_uri='europe-docker.pkg.dev/vertex-ai/prediction/sklearn-cpu.1-5:latest',
                location='europe-west1'
            )
            
        model_upload.wait()
        logging.info(f"Model uploaded to: {model_upload.resource_name}")
        return model_upload
    
    uploaded_model = upload_model()
    vertex_model.uri = uploaded_model.resource_name
    
    # Deploys trained model to Vertex AI Endpoint
    model_deploy = uploaded_model.deploy(
        machine_type='n1-standard-4',
        endpoint=endpoint,
        traffic_split={"0": 100},
        deployed_model_display_name='lolffate',
    )

    vertex_endpoint.uri = model_deploy.resource_name
    logging.info(f"Model deployed to: {model_deploy.resource_name}")