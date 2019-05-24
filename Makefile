build:
    python3 -m grpc_tools.protoc -I. --python_out=. --grpc_python_out=. protos/dish/dish.proto