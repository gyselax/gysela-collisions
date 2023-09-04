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
$ cmake .. -G Ninja -DCMAKE_TOOLCHAIN_FILE=../toolchains/mi250.cce.adastra/toolchain.cmake
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

Note that to run the tests on Unix systems, one may need to run the following
command :
```
$ ulimit -s unlimited
```

This is due to our usage of of stack allocated (constant) buffer representing
the input to the operator. This stack issue could be avoided with the use of
dynamic allocation. The operator itself is not limited by the stack like the
test case is, thus, there is not need to `ulimit` to be called.

## Ideology

Generally, Keeping the interface (It) Simple (and Stupid) (KISS) is a good advice, though never remove necessary complexity (do not make it simpler, stupider that it should). An other view would be to try to use the weakest (less powerful) tool that satisfies the need (it requires that you know what you need, potentially months or years in advance).

## Coding style

<!--
TODO(): [LOOP|LOAD|SIMPLIFY|INDEX|ASYNC|COPY]
NOTE:
-->

## TODOs

- Document the C interface:
    - When to fence.
    - Assuming we CPUs for computation, the input buffer are expected to be on the CPU.
    - Assuming we GPUs for computation, the input buffer are expected to be on the CPU too! `koliop::utility::Copy::RawDataToView<koliop::utility::View::UHL6<Flo64>, koliop::utility::View::MDL6>` vs `koliop::utility::Copy::RawDataToView<koliop::utility::View::UDL6<Flo64>, koliop::utility::View::MDL6>`

- Validation:
    - Test KOKKOS_ENABLE_DEBUG_BOUNDS_CHECK/Kokkos_ENABLE_DEBUG_BOUNDS_CHECK Kokkos_ENABLE_DEBUG

- Tuning:
    - Tune for CPU. NOTE: Should be limited to some layout left/right issues in views or iteration order;
    - Simplify computations;
    - Reduce register usage. Each time you use a view, you consume a LOT of register (dimension/stride storage ?).. much more than a simple pointer;
        - File a bug regarding this issue ?
    - Reduced precision ?:
        - NOTE: that the fact that we do not have pivoting for the LU factorization *could* reduce the precision more when treating Flo32 matrices.
    - ldg (non cached/temporal) loads. There are many views that we read only
    once in a kernel and thus do not need to be cached.

- CMake:
    - https://github.com/cpm-cmake/CPM.cmake
