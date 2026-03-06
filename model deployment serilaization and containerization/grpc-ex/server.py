import grpc
from concurrent import futures
import numpy as np
import joblib
import prediction_pb2
import prediction_pb2_grpc


class MLModelService(prediction_pb2_grpc.MLModelServicer):
    def __init__(self, model_path='model.pkl'):
        """Load the trained model from disk"""
        print(f"Loading model from {model_path}...")
        self.model = joblib.load(model_path)
        self.version = "v1.0"
        print("Model loaded successfully")

    def Predict(self, request, context):
        """Handle prediction requests"""
        try:
            # Convert request to numpy array
            features = np.array(request.features).reshape(1, -1)

            # Make prediction
            prediction = float(self.model.predict(features)[0])

            print(
                f"Prediction request: {request.features} -> {prediction:.2f}")

            # Return response
            return prediction_pb2.PredictResponse(
                prediction=prediction,
                model_version=self.version
            )
        except Exception as e:
            context.set_code(grpc.StatusCode.INTERNAL)
            context.set_details(f"Prediction failed: {str(e)}")
            return prediction_pb2.PredictResponse()


def serve():
    """Start the gRPC server"""
    server = grpc.server(futures.ThreadPoolExecutor(max_workers=10))
    prediction_pb2_grpc.add_MLModelServicer_to_server(
        MLModelService(), server
    )
    server.add_insecure_port('[::]:5000')
    server.start()

    print("Server started on port 5000")
    server.wait_for_termination()


if __name__ == '__main__':
    serve()
