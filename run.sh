#!/bin/bash

HOST_PATH="/Users/weichengtu/Project_git/docker/keras_tf_test/src"

docker run -v ${HOST_PATH}:/app -it dowayching/keras_tf_test bash
