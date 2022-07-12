#!/usr/bin/env bash

# change to project root dir
DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null 2>&1 && pwd )"
cd "$DIR/.."

# patch out parts only needed for live work
cp docker/Dockerfile /tmp/Dockerfile
patch /tmp/Dockerfile scripts/docker_ci.patch

# docker login gitlab.informatik.uni-bremen.de:5005
docker build -f /tmp/Dockerfile -t gitlab.informatik.uni-bremen.de:5005/ascadian/schau_mir_in_die_augen/smida_test .
docker push gitlab.informatik.uni-bremen.de:5005/ascadian/schau_mir_in_die_augen/smida_test
