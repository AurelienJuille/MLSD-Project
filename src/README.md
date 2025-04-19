# Milestone 2

## Sprint 3: API implementation

### Requirements
Back on the project description, we have the following  requirements:
1. Build an API to serve your model
2. Package your model serving API in a Docker container
3. Deploy your model serving API in the Cloud. You should be able to call your model to generate new predictions from another machine.

### 1. Build API to serve the model
This has be done in both `local_app` and `online_app` thanks to the `app.py` file. They allow us to build our flask applications.

To launch the apps local, just do `python ./app.py `

### 2. Package API in a docker container
This has be done in both `local_app` and `online_app` thanks to our `Dockerfile` file.

1. Build the docker image with `docker build -t <image_name> .`
2. Run the image with `docker run -p <host_port>:<container_port> <image_name>`

### 3. Deploy your model serving API in the Cloud
This part has been done manually, the deployed can be found here https://flask-app-30182159501.europe-west1.run.app/.


## Sprint 4: Model pipeline
In this sprint, you have implemented a model pipeline to automatically run different steps of your model training and deployment.

The resulting pipeline construction and components are available in the [pipeline folder](pipeline), with a `Dockerfile` as well.


## Why 2 versions of our model serving API ?
This is because the Riot API that we use only works locally at the moment, therefore 2 versions were created in order to also have a version which works online, to fill the requirements, and one more practical which works locally.
