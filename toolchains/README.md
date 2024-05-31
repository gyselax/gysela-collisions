# Pre-made build settings

## General usage

Prepare the environment, source the environment, use the environment to build:

```
$ mkdir -p build && cd build
$ ../toolchains/mi250.hipcc.adastra/prepare.sh
$ source ../toolchains/mi250.hipcc.adastra/environment.sh
$ cmake .. --toolchain=../toolchains/mi250.hipcc.adastra/toolchain.cmake
```

## Toolchains

The [toolchains](https://en.wikipedia.org/wiki/Toolchain) are represented using a toolchain file. It summarize CMake build settings for the appropriate machine environment and hardware.

On can use the toolchains like so:

```
$ mkdir -p build && cd build
$ cmake .. --toolchain=../toolchains/mi250.hipcc.adastra/toolchain.cmake
```

## Environments

Toolchains may require a non default environment. For that purpose each toolchain will expose the concept of environment script. Each toolchain should provide (a potentially empty) environment file under the `environment.sh` filename. The environment file is used like so:

```
$ source toolchains/mi250.hipcc.adastra/environment.sh
```

The *sourcing* of an environment file will generally happen before the build or usage of the built product.


## Preparing environments

The environment may need preparation (say it is not available on the machine). For that purpose each toolchain will expose the concept of preparation script. Each toolchain should provide (a potentially empty) preparation file under the `prepare.sh` filename. The preparation file is used like so:

```
$ toolchains/mi250.hipcc.adastra/prepare.sh
```

## General notes

- The environment and preparation script should be **working directory agnostic**;
- Ideally, if producing files (such as the installation of a dependency), placing them in the appropriate location such as the WORK file system area often found on HPC machines;
- HPC machines evolves, are notoriously heterogenous, unstable and buggy, you should document issues and workarounds explored as part of the code of the toolchain.
- If the library (say this one) is used by an other project (using `add_subdirectory` or `FetchContent_Declare`), one can not use the library's toolchain as shown above. In this case, it is recommended to modify the parent project so that it contains the appropriate settings present in the library's toolchain file.
