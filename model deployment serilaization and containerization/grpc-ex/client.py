import grpc
import prediction_pb2
import prediction_pb2_grpc


def predict(stub, features):
    """Send prediction request using existing stub"""
    request = prediction_pb2.PredictRequest(features=features)
    response = stub.Predict(request)
    return response


def main():
    """Run prediction examples"""
    test_cases = [
        [1.0],
        [3.5],
        [5.0],
        [10.0]
    ]

    print("Making predictions...\n")

    # Open one channel and stub for all requests
    with grpc.insecure_channel('localhost:500') as channel:
        stub = prediction_pb2_grpc.MLModelStub(channel)

        for features in test_cases:
            try:
                response = predict(stub, features)
                print(f"Input: {features[0]:.1f}")
                print(f"Prediction: {response.prediction:.2f}")
                print(f"Model: {response.model_version}")
                print("-" * 30)
            except grpc.RpcError as e:
                print(f"Error: {e.details()}")


if __name__ == '__main__':
    main()
