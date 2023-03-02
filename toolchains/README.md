# Pre-made build settings

## Toolchains

The toolchains are represented using a toolchain file. It summarize CMake build
settings for the appropriate machine environment, hardware.

On can use the toolchains like so:
```
$ cd koliop && mkdir -p build && cd build
$ cmake .. -DCMAKE_TOOLCHAIN_FILE=../toolchains/acc.cce.adastra/toolchain.cmake
```

If the library (say this one) is used by an other project (using
`add_subdirectory` or `FetchContent_Declare`), one can not use the included
library's toolchain. In this case, it is recommended to modify the parent
project (the client of the library) so that it contains the appropriate settings
present in the library's toolchain file.

## Manual

One can also specify the build settings of the toolchain manually, directly in
the command line:
```
$ cd koliop && mkdir -p build && cd build
$ cmake .. -DCMAKE_CXX_COMPILER="CC" -DKokkos_ENABLE_HIP=NO -Dkoliop_FLAGS_BASE="..." etc...
```
