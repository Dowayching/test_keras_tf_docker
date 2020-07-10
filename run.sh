#!/bin/bash

HOST_PATH="/Users/weichengtu/Project_git/test_keras_tf_docker/src"

docker run -v ${HOST_PATH}:/app -it dowayching/keras_tf_test bash
