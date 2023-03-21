# Koliop

A Kokkos based Collision Operator (KoLiOp) for Gysela.

<!-- koliop (kokkos, Collision, Operator)
gcokos (Gysela, Collision, Kokkos) -->

## Build

### Cloning the repository

```bash
$ git clone <TODO(Etienne M)>.git
```

### Dependencies

```
cmake > 3.16 (3.25 and up is recommended)
```

### Build

If you use [ninja](https://ninja-build.org/), then, under different platforms only the shell changes, not the command. We recommend the use of Ninja but if not possible, make should do as well.

```bash
$ mkdir build && cd build
# Using Ninja:
$ cmake .. -G Ninja -DCMAKE_TOOLCHAIN_FILE=../toolchains/<machine_file>/toolchain.cmake
$ ninja
# Using make:
$ cmake .. -DCMAKE_TOOLCHAIN_FILE=../toolchains/<machine_file>/toolchain.cmake
$ make
```

Where `<machine_file>` being one of the supported machines. For instance:

```bash
$ cmake .. -G Ninja -DCMAKE_TOOLCHAIN_FILE=../toolchains/acc.cce.adastra/toolchain.cmake
```

## Coding style

<!--
TODO(): [LOOP|LOAD|SIMPLIFY|INDEX]
NOTE:
-->
