# gRPC Machine Learning Prediction Service

A simple gRPC-based machine learning prediction service using Python. This project demonstrates how to build and deploy a machine learning model as a gRPC service with a client-server architecture.

## Project Overview

This project contains:

- A simple linear regression model for predictions
- gRPC server that serves the ML model
- gRPC client for making prediction requests
- Protocol buffer definitions for communication
- Docker support for containerization

## Step-by-Step Setup and Usage

### 1. Install Dependencies

```
uv sync
```

and activate the environment:

```
source .venv/bin/activate
# On Windows use:
.venv\Scripts\activate
```

### 2. Train the Machine Learning Model

```
python train.py
```

This will:

- Train a simple linear regression model (y = 2x)
- Save the model to `model.pkl`
- Display model coefficients

### 3. Generate gRPC Code from Protocol Buffer

```
python -m grpc_tools.protoc -I. --python_out=. --grpc_python_out=. prediction.proto
```

This command generates:

- `prediction_pb2.py` - Protocol buffer message classes
- `prediction_pb2_grpc.py` - gRPC service classes

### 4. Build Docker Image

Make sure you have Docker installed and running. Then build the Docker image:

```
docker build -t grpc-server .
```

### 5. Run Docker Container

```
docker run -p 500:5000 grpc-server
```

### 6. Check with gRPC client

In a separate terminal, run the client:

```
python client.py
```

This will send a prediction request to the gRPC server and display the response.
