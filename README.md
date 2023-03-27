# Koliop

A Kokkos based Collision Operator (KoLiOp) for Gysela.

<!-- koliop (kokkos, Collision, Operator)
gcokos (Gysela, Collision, Kokkos) -->

## Building

### Cloning the repository

```bash
$ git clone <TODO(Etienne M)>.git
```

### Dependencies

```
cmake > 3.16 (3.25 and up is recommended)
```

### Compiling

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

## Coding style

<!--
TODO(): [LOOP|LOAD|SIMPLIFY|INDEX]
NOTE:
-->

## TODOs

- Which blas implementation for the LU decomposition
- Port more kernels
- Test KOKKOS_BOUNDS_CHECK
- Create the fortran bindings
- Quick recheck of the kernel and their fortran equivalent
- Test in Gysela (progressively replace libcoll with koliop)

- Simplify operations

- Run on GPU (assume we get GPU views as input)

- Inter exchange loops
- Tune for GPU (coalesced reads)

- Remove reference to std:: in kernels
- Cleanup redundant names, long namespaces, View naming
- Replace reference to Flo64 with something that can be changed more easily. For instance, to turn on/off the use of float instead of double.
- Split the implementation in more files (one per function ?)
- Document the C interface
- Make sure the C interface does not leak (exception et al.)
