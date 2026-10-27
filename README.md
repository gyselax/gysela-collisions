# Koliop

A KOkkos based colLIsion OPerator (KoLiOp) for Gysela that computes the evolution of the distribution function due to collisions. Some documents can be found in `/documents`.

![KOLIOP architecture](documents/KOLIOP_Architecture.png "KOLIOP Architecture")

## Building

### Cloning the repository

```bash
$ git clone https://gitlab.com/cines/code.gysela/libkoliop
```

### Dependencies

```
cmake >= 3.22.1 (3.25 and up is recommended)
```

### Compiling

If you use [ninja](https://ninja-build.org/), then, under different platforms only the shell changes, not the command. We recommend the use of Ninja but if not possible, make should do as well.

```bash
$ mkdir build && cd build
# Using Ninja:
$ cmake .. -GNinja --toolchain=../toolchains/<machine_file>/toolchain.cmake
$ ninja
# Using make:
$ cmake .. --toolchain=../toolchains/<machine_file>/toolchain.cmake
$ make
```

Where `<machine_file>` being one of the supported machines. For instance:

```bash
$ cmake .. -GNinja --toolchain=../toolchains/mi250.cce.adastra/toolchain.cmake
```

### Testing

Simply run the following command in the build directory (typically `build`).
```
$ ctest
Test project /lus/home/BCINES/dci/malaboeuf/repos/gysela-gpu-porting/koliop/build
    Start 1: test_dummy_test
1/1 Test #1: test_dummy_test ..................   Passed    0.61 sec

100% tests passed, 0 tests failed out of 1

Total Test time (real) =   0.63 sec
```

