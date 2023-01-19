#!/bin/sh
set -e

echo "---- Check shell scripts styling with ShellCheck ----"

docker pull mvassilev/alpine-shellcheck
alias shellcheck='docker run --rm -it -v $(pwd):/mnt mvassilev/alpine-shellcheck'

shellcheck --version
shellcheck  **/*.sh                     \
            onbuild/auto_update_hosts   \
            onbuild/get_hosts           \
            onbuild/mpi_bootstrap       \

echo "=> No styling trouble found"
