# Koliop

A Kokkos based Collision Operator (KoLiOp) for Gysela.

koliop (kokkos, Collision, Operator)
gcokos (Gysela, Collision, Kokkos)

### Version constraints

```
TODO(Etienne M):
```

## Cloning the repository

1. Clone the repository
```bash
$ git clone <TODO(Etienne M)>.git
```
2. Install & Update the required packages. Please refer to the `Build` section.

## Build

Under different platforms only the shell changes, not the command:
```
$ mkdir build && cd build
$ cmake .. -DCMAKE_TOOLCHAIN_FILE=../toolchains/<machine_file>/toolchain.cmake
$ make
```

Note that you are recommended to not use make but prefer a tool such as [ninja](https://ninja-build.org/).

<!--
TODO(): [LOOP|LOAD|SIMPLIFY|INDEX]
NOTE:
-->
