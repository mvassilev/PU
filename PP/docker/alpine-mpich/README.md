# Alpine MPICH

Docker image of Alpine Linux with  [MPICH](http://www.mpich.org/) -- portable implementation of Message Passing Interface (MPI) standard. Designed for MPI program development and deployment.

Provide solution for MPI Cluster Automation with Docker containers using either Docker Compose or Docker Swarm Mode.

----

Image usage instruction: 
[https://hub.docker.com/r/mvassilev/alpine-mpich](https://hub.docker.com/r/mvassilev/alpine-mpich)


Distributed MPI cluster setup instruction: [https://github.com/mvassilev/alpine-mpich/tree/master/cluster](https://github.com/mvassilev/alpine-mpich/tree/master/cluster)  
* **[Single Host Orchestration](https://github.com/mvassilev/PU/wiki/Single-Host-Orchestration)**
* **[Multi Host Orchestration](https://github.com/mvassilev/PU/wiki/Multi-Host-Orchestration)**


----

`base image` ([Dockerfile](https://github.com/mvassilev/alpine-mpich/blob/master/Dockerfile)) : contains MPICH and essential build tools. Intended to be used as development environment for developing MPI programs.

`onbuild image` ([Dockerfile](https://github.com/mvassilev/alpine-mpich/blob/onbuild/Dockerfile)) : inherits base image with network setup for cluster. Can be used like base image but intended to be used to build image that contains compiled MPI program in order to deploy to a cluster.

`cluster` ([project scaffolder](https://github.com/mvassilev/alpine-mpich/tree/master/cluster)) : is a directory containing a setup for deploying MPI programs to a cluster of containers. Include a runner script to automate Docker commands.


*Below is instruction for building the Docker image yourself if you don't want to use the pre-built base or onbuild image.*

----

## Build Instruction

The images are prebuilt and hosted at Docker Hub, but in case you want to build them yourself:

```sh
$ git clone https://github.com/mvassilev/alpine-mpich

$ cd alpine-mpich

$ docker build -t mvassilev/alpine-mpich:base base/

$ docker build -t mvassilev/alpine-mpich:onbuild onbuild/
```

Since the onbuild image inherits the base image, if you use a different tag name (`mvassilev/alpine-mpich`), you must change the first line in `onbuild/Dockerfile` to inherits `FROM` your custom tag name.

----

## Build Customization

In order to customize the base image at build time, you need to download the Dockerfile source code and build with optional build arguments (without those, you get the exact image as you pull from DockerHub), for example:

```sh
$ git clone https://github.com/mvassilev/alpine-mpich

$ cd alpine-mpich

$ docker build --build-arg MPICH_VERSION="3.2b4" -t my-custom-image base/
```

These are available **build arguments** to customize the build:
- `REQUIRE` *space-separated names of packages to be installed from Alpine main [package repository](http://pkgs.alpinelinux.org/packages) before downloading and installing MPICH. Default=`"sudo build-base openssh"`*
- `MPICH_VERSION` *to find which version of MPICH to download from [here](http://www.mpich.org/static/downloads/). Default=`"3.2"`*
- `MPICH_CONFIGURE_OPTIONS` *to be passed to `./configure` in MPICH source directory. Default=`"--disable-fortran"`* (let me know if you need Fortran, I can certainly add it by default)
- `MPICH_MAKE_OPTIONS` *to be passed to `make` after the above command. Default is empty*
- `USER` *non-root user with sudo privilege and no password required. Default=`mpi`*
- `WORKDIR` *main working directory to be owned by default user. Default=`/project`*

*See MPICH documentation for available options*

Should you need more than that, you need to change the Dockerfile yourself or send suggestion/pull requests to this GitHub repository.


# Feedback

Feedbacks are always welcome. For general comments, use the comment section at the bottom of this [image page](https://hub.docker.com/r/mvassilev/alpine-mpich) on Docker Hub

## Issue

Use this GitHub repository [issues](https://github.com/mvassilev/alpine-mpich/issues)

## Contributing

Suggestions and pull requests are awesome.

# License MIT
Spacial thanks to [Nikyle Nguyen](https://github.com/NLKNguyen/alpine-mpich)