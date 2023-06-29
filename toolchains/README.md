# Pre-made build settings

## Toolchains

The [toolchains](https://en.wikipedia.org/wiki/Toolchain) are represented using
a toolchain file. It summarize CMake build settings for the appropriate machine
environment and hardware.

On can use the toolchains like so:
```
$ cd koliop && mkdir -p build && cd build
$ cmake .. -DCMAKE_TOOLCHAIN_FILE=../toolchains/mi250.cce.adastra/toolchain.cmake
```

If the library (say this one) is used by an other project (using
`add_subdirectory` or `FetchContent_Declare`), one can not use the library's
toolchain as shown above. In this case, it is recommended to modify the parent
project so that it contains the appropriate settings present in the library's
toolchain file. The parent project would refer to the client of the library.


## Manual

One can also specify the build settings of the toolchain manually, directly in
the command line:
```
$ cd koliop && mkdir -p build && cd build
$ cmake .. -DCMAKE_CXX_COMPILER="CC" -DKokkos_ENABLE_HIP=NO -Dkoliop_FLAGS_BASE="..." etc...
```
