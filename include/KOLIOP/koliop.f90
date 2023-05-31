! SPDX-License-Identifier: MIT

MODULE koliop

   USE ISO_C_Binding

   IMPLICIT NONE

   PRIVATE

   PUBLIC :: koliop_Status

   PUBLIC :: KOLIOP_STATUS_SUCCESS
   PUBLIC :: KOLIOP_STATUS_UNKOWN_FAILURE
   PUBLIC :: KOLIOP_STATUS_INITIALIZATION_FAILURE
   PUBLIC :: KOLIOP_STATUS_DEINITIALIZATION_FAILURE
   PUBLIC :: KOLIOP_STATUS_INVALID_VALUE

   PUBLIC :: koliop_Initialize_f
   PUBLIC :: koliop_Deinitialize_f
   PUBLIC :: koliop_Create_f
   PUBLIC :: koliop_Release_f
   PUBLIC :: koliop_ComputeMoments_f
   PUBLIC :: koliop_PrecomputeNu_f
   PUBLIC :: koliop_ComputeMaxwellianEvolution_f
   PUBLIC :: koliop_ComputeQparOnNT_f
   PUBLIC :: koliop_ComputeCpar_f
   PUBLIC :: koliop_ComputeMaxwellianUnshiftedLoc_f
   PUBLIC :: koliop_Apply_f
   PUBLIC :: koliop_ComputeDensityCorrection_f
   PUBLIC :: koliop_ComputeUparTCorrection_f

   INTEGER, PARAMETER :: koliop_Status = C_INT32_T

   INTEGER(koliop_Status), PARAMETER :: KOLIOP_STATUS_SUCCESS = 0
   INTEGER(koliop_Status), PARAMETER :: KOLIOP_STATUS_UNKOWN_FAILURE = -1
   INTEGER(koliop_Status), PARAMETER :: KOLIOP_STATUS_INITIALIZATION_FAILURE = -2
   INTEGER(koliop_Status), PARAMETER :: KOLIOP_STATUS_DEINITIALIZATION_FAILURE = -3
   INTEGER(koliop_Status), PARAMETER :: KOLIOP_STATUS_INVALID_VALUE = -30

   ! NOTE: Sources:
   ! https://www.cism.ucl.ac.be/Services/Formations/ICS/ics_2013.0.028/composer_xe_2013.1.117/Documentation/en_US/compiler_f/main_for/GUID-ECF6FF36-1397-43D8-88BB-CCD2F6108DA5.htm
   ! https://docs.oracle.com/cd/E19422-01/819-3685/11_cfort.html
   ! https://gcc.gnu.org/onlinedocs/gfortran/ISO_005fC_005fBINDING.html
   ! https://en.cppreference.com/w/c/language/type
   ! https://gcc.gnu.org/onlinedocs/gcc-3.4.6/g77/Table-of-Intrinsic-Functions.html
   ! https://community.intel.com/t5/Intel-Fortran-Compiler/How-to-pass-an-allocated-array-to-C/td-p/1109471
   ! https://www.ibm.com/docs/en/xl-fortran-linux/16.1.1?topic=attributes-target
   !
   ! We use: TYPE(C_PTR), VALUE to pass arrays to the C code. We could hav used
   ! REAL(C_DOUBLE), DIMENSION(*)
   !
   INTERFACE
      ! koliop_Status koliop_Initialize(size_t a_host_thread_count,
      !                                 size_t a_device_index,
      !                                 int8_t with_verbose_initialization);
      INTEGER(koliop_Status) FUNCTION koliop_Initialize_c( &
         a_host_thread_count, &
         a_device_index, &
         with_verbose_initialization) &
         BIND(C, NAME='koliop_Initialize')

         IMPORT :: koliop_Status
         IMPORT :: C_INT8_T
         IMPORT :: C_SIZE_T

         IMPLICIT NONE

         INTEGER(C_SIZE_T), VALUE, INTENT(IN) :: a_host_thread_count
         INTEGER(C_SIZE_T), VALUE, INTENT(IN) :: a_device_index
         INTEGER(C_INT8_T), VALUE, INTENT(IN) :: with_verbose_initialization
      END FUNCTION

      ! koliop_Status koliop_Deinitialize(void);
      INTEGER(koliop_Status) FUNCTION koliop_Deinitialize_c() &
         BIND(C, NAME='koliop_Deinitialize')

         IMPORT :: koliop_Status

         IMPLICIT NONE
      END FUNCTION

      ! koliop_Status koliop_Create(koliop_Operator* an_operator_handle,
      !                             size_t           Npolmax,
      !                             size_t           index_max,
      !                             int64_t          pSpecies_id,
      !                             int64_t          mu_id,
      !                             size_t           the_mu_extent,
      !                             size_t           the_vpar_extent,
      !                             size_t           the_r_extent,
      !                             size_t           the_theta_extent,
      !                             size_t           the_phi_extent,
      !                             size_t           the_species_extent,
      !                             double           R0,
      !                             int8_t           collision_interspecies,
      !                             int8_t           SOL,
      !                             int8_t           LIM,
      !                             int64_t          ir_SOL_separatrix,
      !                             double           dt,
      !                             double           dvpar,
      !                             const double*    comb_mat,
      !                             const double*    hat_As,
      !                             const double*    hat_Zs,
      !                             const double*    mug,
      !                             const double*    vparg,
      !                             const double*    coeff_intdmu,
      !                             const double*    coeff_intdvpar,
      !                             const double*    rg,
      !                             const double*    nustar0_r,
      !                             const double*    q_r,
      !                             const double*    mask_buffer_r,
      !                             const double*    mask_LIM,
      !                             const double*    B_norm,
      !                             const double*    Bstar_s);
      INTEGER(koliop_Status) FUNCTION koliop_Create_c( &
         an_operator_handle, &
         Npolmax, &
         index_max, &
         pSpecies_id, &
         mu_id, &
         the_mu_extent, &
         the_vpar_extent, &
         the_r_extent, &
         the_theta_extent, &
         the_phi_extent, &
         the_species_extent, &
         R0, &
         collision_interspecies, &
         SOL, &
         LIM, &
         ir_SOL_separatrix, &
         dt, &
         dvpar, &
         comb_mat, &
         hat_As, &
         hat_Zs, &
         mug, &
         vparg, &
         coeff_intdmu, &
         coeff_intdvpar, &
         rg, &
         nustar0_r, &
         q_r, &
         mask_buffer_r, &
         mask_LIM, &
         B_norm, &
         Bstar_s) &
         BIND(C, NAME='koliop_Create')

         IMPORT :: koliop_Status
         IMPORT :: C_INT8_T
         IMPORT :: C_INT64_T
         IMPORT :: C_SIZE_T
         IMPORT :: C_DOUBLE
         IMPORT :: C_PTR

         IMPLICIT NONE

         TYPE(C_PTR), INTENT(INOUT) :: an_operator_handle ! NOTE: the C code expects a pointer to a pointer.
         INTEGER(C_SIZE_T), VALUE, INTENT(IN) :: Npolmax
         INTEGER(C_SIZE_T), VALUE, INTENT(IN) :: index_max
         INTEGER(C_INT64_T), VALUE, INTENT(IN) :: pSpecies_id
         INTEGER(C_INT64_T), VALUE, INTENT(IN) :: mu_id
         INTEGER(C_SIZE_T), VALUE, INTENT(IN) :: the_mu_extent
         INTEGER(C_SIZE_T), VALUE, INTENT(IN) :: the_vpar_extent
         INTEGER(C_SIZE_T), VALUE, INTENT(IN) :: the_r_extent
         INTEGER(C_SIZE_T), VALUE, INTENT(IN) :: the_theta_extent
         INTEGER(C_SIZE_T), VALUE, INTENT(IN) :: the_phi_extent
         INTEGER(C_SIZE_T), VALUE, INTENT(IN) :: the_species_extent
         REAL(C_DOUBLE), VALUE, INTENT(IN) :: R0
         INTEGER(C_INT8_T), VALUE, INTENT(IN) :: collision_interspecies
         INTEGER(C_INT8_T), VALUE, INTENT(IN) :: SOL
         INTEGER(C_INT8_T), VALUE, INTENT(IN) :: LIM
         INTEGER(C_INT64_T), VALUE, INTENT(IN) :: ir_SOL_separatrix
         REAL(C_DOUBLE), VALUE, INTENT(IN) :: dt
         REAL(C_DOUBLE), VALUE, INTENT(IN) :: dvpar
         TYPE(C_PTR), VALUE, INTENT(IN) :: comb_mat
         TYPE(C_PTR), VALUE, INTENT(IN) :: hat_As
         TYPE(C_PTR), VALUE, INTENT(IN) :: hat_Zs
         TYPE(C_PTR), VALUE, INTENT(IN) :: mug
         TYPE(C_PTR), VALUE, INTENT(IN) :: vparg
         TYPE(C_PTR), VALUE, INTENT(IN) :: coeff_intdmu
         TYPE(C_PTR), VALUE, INTENT(IN) :: coeff_intdvpar
         TYPE(C_PTR), VALUE, INTENT(IN) :: rg
         TYPE(C_PTR), VALUE, INTENT(IN) :: nustar0_r
         TYPE(C_PTR), VALUE, INTENT(IN) :: q_r
         TYPE(C_PTR), VALUE, INTENT(IN) :: mask_buffer_r
         TYPE(C_PTR), VALUE, INTENT(IN) :: mask_LIM
         TYPE(C_PTR), VALUE, INTENT(IN) :: B_norm
         TYPE(C_PTR), VALUE, INTENT(IN) :: Bstar_s
      END FUNCTION

      ! koliop_Status koliop_Release(koliop_Operator an_operator)
      INTEGER(koliop_Status) FUNCTION koliop_Release_c( &
         an_operator) &
         BIND(C, NAME='koliop_Release')

         IMPORT :: koliop_Status
         IMPORT :: C_PTR

         IMPLICIT NONE

         TYPE(C_PTR), VALUE, INTENT(IN) :: an_operator
      END FUNCTION

      ! koliop_Status koliop_ComputeMoments(koliop_Operator an_operator,
      !                                     double*         f5d_s)
      INTEGER(koliop_Status) FUNCTION koliop_ComputeMoments_c( &
         an_operator, &
         f5d_s) &
         BIND(C, NAME='koliop_ComputeMoments')

         IMPORT :: koliop_Status
         IMPORT :: C_PTR

         IMPLICIT NONE

         TYPE(C_PTR), VALUE, INTENT(IN) :: an_operator
         TYPE(C_PTR), VALUE, INTENT(IN) :: f5d_s
      END FUNCTION

      ! koliop_Status koliop_PrecomputeNu(koliop_Operator an_operator);
      INTEGER(koliop_Status) FUNCTION koliop_PrecomputeNu_c( &
         an_operator) &
         BIND(C, NAME='koliop_PrecomputeNu')

         IMPORT :: koliop_Status
         IMPORT :: C_PTR

         IMPLICIT NONE

         TYPE(C_PTR), VALUE, INTENT(IN) :: an_operator
      END FUNCTION

      ! koliop_Status koliop_ComputeMaxwellianEvolution(koliop_Operator an_operator,
      !                                                 double*         f5d_s);
      INTEGER(koliop_Status) FUNCTION koliop_ComputeMaxwellianEvolution_c( &
         an_operator, &
         f5d_s) &
         BIND(C, NAME='koliop_ComputeMaxwellianEvolution')

         IMPORT :: koliop_Status
         IMPORT :: C_PTR

         IMPLICIT NONE

         TYPE(C_PTR), VALUE, INTENT(IN) :: an_operator
         TYPE(C_PTR), VALUE, INTENT(IN) :: f5d_s
      END FUNCTION

      ! koliop_Status koliop_ComputeQparOnNT(koliop_Operator an_operator,
      !                                      const double*   f5d_s);
      INTEGER(koliop_Status) FUNCTION koliop_ComputeQparOnNT_c( &
         an_operator, &
         f5d_s) &
         BIND(C, NAME='koliop_ComputeQparOnNT')

         IMPORT :: koliop_Status
         IMPORT :: C_PTR

         IMPLICIT NONE

         TYPE(C_PTR), VALUE, INTENT(IN) :: an_operator
         TYPE(C_PTR), VALUE, INTENT(IN) :: f5d_s
      END FUNCTION

      ! koliop_Status koliop_ComputeCpar(koliop_Operator an_operator,
      !                                  double*         f5d_s);
      INTEGER(koliop_Status) FUNCTION koliop_ComputeCpar_c( &
         an_operator, &
         f5d_s) &
         BIND(C, NAME='koliop_ComputeCpar')

         IMPORT :: koliop_Status
         IMPORT :: C_PTR

         IMPLICIT NONE

         TYPE(C_PTR), VALUE, INTENT(IN) :: an_operator
         TYPE(C_PTR), VALUE, INTENT(IN) :: f5d_s
      END FUNCTION

      ! koliop_Status koliop_ComputeMaxwellianUnshiftedLoc(koliop_Operator an_operator,
      !                                                    double*         f5d_s,
      !                                                    int32_t         choice);
      INTEGER(koliop_Status) FUNCTION koliop_ComputeMaxwellianUnshiftedLoc_c( &
         an_operator, &
         f5d_s, &
         choice) &
         BIND(C, NAME='koliop_ComputeMaxwellianUnshiftedLoc')

         IMPORT :: koliop_Status
         IMPORT :: C_INT32_T
         IMPORT :: C_PTR

         IMPLICIT NONE

         TYPE(C_PTR), VALUE, INTENT(IN) :: an_operator
         TYPE(C_PTR), VALUE, INTENT(IN) :: f5d_s
         INTEGER(C_INT32_T), VALUE, INTENT(IN) :: choice
      END FUNCTION

      ! koliop_Status koliop_Apply(koliop_Operator an_operator,
      !                            double*         f5d_s);
      INTEGER(koliop_Status) FUNCTION koliop_Apply_c( &
         an_operator, &
         f5d_s) &
         BIND(C, NAME='koliop_Apply')

         IMPORT :: koliop_Status
         IMPORT :: C_PTR

         IMPLICIT NONE

         TYPE(C_PTR), VALUE, INTENT(IN) :: an_operator
         TYPE(C_PTR), VALUE, INTENT(IN) :: f5d_s
      END FUNCTION

      ! koliop_Status koliop_ComputeDensityCorrection(koliop_Operator an_operator,
      !                                               double*         f5d_s);
      INTEGER(koliop_Status) FUNCTION koliop_ComputeDensityCorrection_c( &
         an_operator, &
         f5d_s) &
         BIND(C, NAME='koliop_ComputeDensityCorrection')

         IMPORT :: koliop_Status
         IMPORT :: C_PTR

         IMPLICIT NONE

         TYPE(C_PTR), VALUE, INTENT(IN) :: an_operator
         TYPE(C_PTR), VALUE, INTENT(IN) :: f5d_s
      END FUNCTION

      ! koliop_Status koliop_ComputeUparTCorrection(koliop_Operator an_operator,
      !                                             double*         f5d_s);
      INTEGER(koliop_Status) FUNCTION koliop_ComputeUparTCorrection_c( &
         an_operator, &
         f5d_s) &
         BIND(C, NAME='koliop_ComputeUparTCorrection')

         IMPORT :: koliop_Status
         IMPORT :: C_PTR

         IMPLICIT NONE

         TYPE(C_PTR), VALUE, INTENT(IN) :: an_operator
         TYPE(C_PTR), VALUE, INTENT(IN) :: f5d_s
      END FUNCTION
   END INTERFACE

CONTAINS

   SUBROUTINE koliop_Initialize_f( &
      a_host_thread_count, &
      a_device_index, &
      with_verbose_initialization, &
      an_error_code)

      IMPLICIT NONE

      INTEGER, INTENT(IN) :: a_host_thread_count
      INTEGER, INTENT(IN) :: a_device_index
      LOGICAL, INTENT(IN) :: with_verbose_initialization
      INTEGER(koliop_Status), INTENT(OUT) :: an_error_code

      INTEGER(C_INT8_T) :: with_verbose_initialization_cast

      IF (with_verbose_initialization) THEN
         with_verbose_initialization_cast = 1
      ELSE
         with_verbose_initialization_cast = 0
      END IF

      an_error_code = koliop_Initialize_c( &
         INT(a_host_thread_count, C_SIZE_T), &
         INT(a_device_index, C_SIZE_T), &
         with_verbose_initialization_cast)
   END SUBROUTINE

   SUBROUTINE koliop_Deinitialize_f( &
      an_error_code)

      IMPLICIT NONE

      INTEGER(koliop_Status), INTENT(OUT) :: an_error_code

      an_error_code = koliop_Deinitialize_c()
   END SUBROUTINE

   SUBROUTINE koliop_Create_f(&
      an_operator_handle, &
      Npolmax, &
      index_max, &
      pSpecies_id, &
      mu_id, &
      the_mu_extent, &
      the_vpar_extent, &
      the_r_extent, &
      the_theta_extent, &
      the_phi_extent, &
      the_species_extent, &
      R0, &
      collision_interspecies, &
      SOL, &
      LIM, &
      ir_SOL_separatrix, &
      dt, &
      dvpar, &
      comb_mat, &
      hat_As, &
      hat_Zs, &
      mug, &
      vparg, &
      coeff_intdmu, &
      coeff_intdvpar, &
      rg, &
      nustar0_r, &
      q_r, &
      mask_buffer_r, &
      mask_LIM, &
      B_norm, &
      Bstar_s, &
      an_error_code)

      IMPLICIT NONE

      TYPE(C_PTR), INTENT(INOUT) :: an_operator_handle
      INTEGER, INTENT(IN) :: Npolmax
      INTEGER, INTENT(IN) :: index_max
      INTEGER, INTENT(IN) :: pSpecies_id
      INTEGER, INTENT(IN) :: mu_id
      INTEGER, INTENT(IN) :: the_mu_extent
      INTEGER, INTENT(IN) :: the_vpar_extent
      INTEGER, INTENT(IN) :: the_r_extent
      INTEGER, INTENT(IN) :: the_theta_extent
      INTEGER, INTENT(IN) :: the_phi_extent
      INTEGER, INTENT(IN) :: the_species_extent
      REAL(C_DOUBLE), INTENT(IN) :: R0
      LOGICAL, INTENT(IN) :: collision_interspecies
      LOGICAL, INTENT(IN) :: SOL
      LOGICAL, INTENT(IN) :: LIM
      INTEGER, INTENT(IN) :: ir_SOL_separatrix
      REAL(C_DOUBLE), INTENT(IN) :: dt
      REAL(C_DOUBLE), INTENT(IN) :: dvpar
      REAL(C_DOUBLE), TARGET, DIMENSION(*), INTENT(IN) :: comb_mat
      REAL(C_DOUBLE), TARGET, DIMENSION(*), INTENT(IN) :: hat_As
      REAL(C_DOUBLE), TARGET, DIMENSION(*), INTENT(IN) :: hat_Zs
      REAL(C_DOUBLE), TARGET, DIMENSION(*), INTENT(IN) :: mug
      REAL(C_DOUBLE), TARGET, DIMENSION(*), INTENT(IN) :: vparg
      REAL(C_DOUBLE), TARGET, DIMENSION(*), INTENT(IN) :: coeff_intdmu
      REAL(C_DOUBLE), TARGET, DIMENSION(*), INTENT(IN) :: coeff_intdvpar
      REAL(C_DOUBLE), TARGET, DIMENSION(*), INTENT(IN) :: rg
      REAL(C_DOUBLE), TARGET, DIMENSION(*), INTENT(IN) :: nustar0_r
      REAL(C_DOUBLE), TARGET, DIMENSION(*), INTENT(IN) :: q_r
      REAL(C_DOUBLE), TARGET, DIMENSION(*), INTENT(IN) :: mask_buffer_r
      REAL(C_DOUBLE), TARGET, DIMENSION(*), INTENT(IN) :: mask_LIM
      REAL(C_DOUBLE), TARGET, DIMENSION(*), INTENT(IN) :: B_norm
      REAL(C_DOUBLE), TARGET, DIMENSION(*), INTENT(IN) :: Bstar_s
      INTEGER(koliop_Status), INTENT(OUT) :: an_error_code

      INTEGER(C_INT8_T) :: collision_interspecies_cast
      INTEGER(C_INT8_T) :: SOL_cast
      INTEGER(C_INT8_T) :: LIM_cast

      IF (collision_interspecies) THEN
         collision_interspecies_cast = 1
      ELSE
         collision_interspecies_cast = 0
      END IF

      IF (SOL) THEN
         SOL_cast = 1
      ELSE
         SOL_cast = 0
      END IF

      IF (LIM) THEN
         LIM_cast = 1
      ELSE
         LIM_cast = 0
      END IF

      an_error_code = koliop_Create_c( &
         an_operator_handle, &
         INT(Npolmax, C_SIZE_T), &
         INT(index_max, C_SIZE_T), &
         INT(pSpecies_id, C_INT64_T), &
         INT(mu_id, C_INT64_T), &
         INT(the_mu_extent, C_SIZE_T), &
         INT(the_vpar_extent, C_SIZE_T), &
         INT(the_r_extent, C_SIZE_T), &
         INT(the_theta_extent, C_SIZE_T), &
         INT(the_phi_extent, C_SIZE_T), &
         INT(the_species_extent, C_SIZE_T), &
         R0, &
         collision_interspecies_cast, &
         SOL_cast, &
         LIM_cast, &
         INT(ir_SOL_separatrix, C_INT64_T), &
         dt, &
         dvpar, &
         C_LOC(comb_mat), &
         C_LOC(hat_As), &
         C_LOC(hat_Zs), &
         C_LOC(mug), &
         C_LOC(vparg), &
         C_LOC(coeff_intdmu), &
         C_LOC(coeff_intdvpar), &
         C_LOC(rg), &
         C_LOC(nustar0_r), &
         C_LOC(q_r), &
         C_LOC(mask_buffer_r), &
         C_LOC(mask_LIM), &
         C_LOC(B_norm), &
         C_LOC(Bstar_s))
   END SUBROUTINE

   SUBROUTINE koliop_Release_f( &
      an_operator_handle, &
      an_error_code)

      IMPLICIT NONE

      TYPE(C_PTR), INTENT(IN) :: an_operator_handle
      INTEGER(koliop_Status), INTENT(OUT) :: an_error_code

      an_error_code = koliop_Release_c(an_operator_handle)
   END SUBROUTINE

   SUBROUTINE koliop_ComputeMoments_f( &
      an_operator_handle, &
      f5d_s, &
      an_error_code)

      IMPLICIT NONE

      TYPE(C_PTR), INTENT(IN) :: an_operator_handle
      REAL(C_DOUBLE), TARGET, DIMENSION(*), INTENT(INOUT) :: f5d_s
      INTEGER(koliop_Status), INTENT(OUT) :: an_error_code

      an_error_code = koliop_ComputeMoments_c( &
         an_operator_handle, &
         C_LOC(f5d_s))
   END SUBROUTINE

   SUBROUTINE koliop_PrecomputeNu_f( &
      an_operator_handle, &
      an_error_code)

      IMPLICIT NONE

      TYPE(C_PTR), INTENT(IN) :: an_operator_handle
      INTEGER(koliop_Status), INTENT(OUT) :: an_error_code

      an_error_code = koliop_PrecomputeNu_c( &
         an_operator_handle)
   END SUBROUTINE

   SUBROUTINE koliop_ComputeMaxwellianEvolution_f( &
      an_operator_handle, &
      f5d_s, &
      an_error_code)

      IMPLICIT NONE

      TYPE(C_PTR), INTENT(IN) :: an_operator_handle
      REAL(C_DOUBLE), TARGET, DIMENSION(*), INTENT(INOUT) :: f5d_s
      INTEGER(koliop_Status), INTENT(OUT) :: an_error_code

      an_error_code = koliop_ComputeMaxwellianEvolution_c( &
         an_operator_handle, &
         C_LOC(f5d_s))
   END SUBROUTINE

   SUBROUTINE koliop_ComputeQparOnNT_f( &
      an_operator_handle, &
      f5d_s, &
      an_error_code)

      IMPLICIT NONE

      TYPE(C_PTR), INTENT(IN) :: an_operator_handle
      REAL(C_DOUBLE), TARGET, DIMENSION(*), INTENT(IN) :: f5d_s
      INTEGER(koliop_Status), INTENT(OUT) :: an_error_code

      an_error_code = koliop_ComputeQparOnNT_c( &
         an_operator_handle, &
         C_LOC(f5d_s))
   END SUBROUTINE

   SUBROUTINE koliop_ComputeCpar_f( &
      an_operator_handle, &
      f5d_s, &
      an_error_code)

      IMPLICIT NONE

      TYPE(C_PTR), INTENT(IN) :: an_operator_handle
      REAL(C_DOUBLE), TARGET, DIMENSION(*), INTENT(INOUT) :: f5d_s
      INTEGER(koliop_Status), INTENT(OUT) :: an_error_code

      an_error_code = koliop_ComputeCpar_c( &
         an_operator_handle, &
         C_LOC(f5d_s))
   END SUBROUTINE

   SUBROUTINE koliop_ComputeMaxwellianUnshiftedLoc_f( &
      an_operator_handle, &
      f5d_s, &
      choice, &
      an_error_code)

      IMPLICIT NONE

      TYPE(C_PTR), INTENT(IN) :: an_operator_handle
      REAL(C_DOUBLE), TARGET, DIMENSION(*), INTENT(INOUT) :: f5d_s
      INTEGER, INTENT(IN) :: choice
      INTEGER(koliop_Status), INTENT(OUT) :: an_error_code

      an_error_code = koliop_ComputeMaxwellianUnshiftedLoc_c( &
         an_operator_handle, &
         C_LOC(f5d_s), &
         INT(choice, C_INT32_T))
   END SUBROUTINE

   SUBROUTINE koliop_Apply_f( &
      an_operator_handle, &
      f5d_s, &
      an_error_code)

      IMPLICIT NONE

      TYPE(C_PTR), INTENT(IN) :: an_operator_handle
      REAL(C_DOUBLE), TARGET, DIMENSION(*), INTENT(INOUT) :: f5d_s
      INTEGER(koliop_Status), INTENT(OUT) :: an_error_code

      an_error_code = koliop_Apply_c( &
         an_operator_handle, &
         C_LOC(f5d_s))
   END SUBROUTINE

   SUBROUTINE koliop_ComputeDensityCorrection_f( &
      an_operator_handle, &
      f5d_s, &
      an_error_code)

      IMPLICIT NONE

      TYPE(C_PTR), INTENT(IN) :: an_operator_handle
      REAL(C_DOUBLE), TARGET, DIMENSION(*), INTENT(INOUT) :: f5d_s
      INTEGER(koliop_Status), INTENT(OUT) :: an_error_code

      an_error_code = koliop_ComputeDensityCorrection_c( &
         an_operator_handle, &
         C_LOC(f5d_s))
   END SUBROUTINE

   SUBROUTINE koliop_ComputeUparTCorrection_f( &
      an_operator_handle, &
      f5d_s, &
      an_error_code)

      IMPLICIT NONE

      TYPE(C_PTR), INTENT(IN) :: an_operator_handle
      REAL(C_DOUBLE), TARGET, DIMENSION(*), INTENT(INOUT) :: f5d_s
      INTEGER(koliop_Status), INTENT(OUT) :: an_error_code

      an_error_code = koliop_ComputeUparTCorrection_c( &
         an_operator_handle, &
         C_LOC(f5d_s))
   END SUBROUTINE
END MODULE

! To test the interface, uncomment the PROGRAM below and build using something
! like:
! ninja && ftn -v -m 2 -O3 ../include/KOLIOP/koliop.f90 -c && cc -v -o test_dummy_test_f koliop.o -Wl,-rpath,/opt/rocm-5.3.0/lib:/opt/rocm-5.3.0/hip/lib libkoliop.a  _deps/kokkoskernels-build/libkokkoskernels.a _deps/kokkos-build/containers/src/libkokkoscontainers.a _deps/kokkos-build/core/src/libkokkoscore.a -ldl /opt/rocm-5.3.0/lib/libhsa-runtime64.so /opt/rocm-5.3.0/hip/lib/libamdhip64.so /opt/rocm-5.3.0/llvm/lib/clang/15.0.0/lib/linux/libclang_rt.builtins-x86_64.a /opt/cray/pe/libsci/22.11.1.2/CRAY/9.0/x86_64/lib/libsci_cray_mpi_mp.so /opt/cray/pe/libsci/22.11.1.2/CRAY/9.0/x86_64/lib/libsci_cray_mp.so /opt/cray/pe/cce/15.0.0/cce/x86_64/lib/libcraymp.so _deps/kokkos-build/simd/src/libkokkossimd.a && echo OK

! PROGRAM Testing
!    USE koliop
!    USE ISO_C_Binding

!    IMPLICIT NONE

!    call koliop_test_Run()

! CONTAINS

!    SUBROUTINE koliop_test_Run()

!       IMPLICIT NONE

!       INTEGER :: a_host_thread_count = 8
!       INTEGER :: a_device_index = 0
!       LOGICAL :: with_verbose_initialization = .TRUE.

!       INTEGER :: Npolmax = 3
!       INTEGER :: index_max = 4
!       INTEGER :: pSpecies_id = 0
!       INTEGER :: mu_id = 4
!       INTEGER :: the_mu_extent = 48
!       INTEGER :: the_vpar_extent = 128
!       INTEGER :: the_r_extent = 32
!       INTEGER :: the_theta_extent = 32
!       INTEGER :: the_phi_extent = 1
!       INTEGER :: the_species_extent = 1
!       REAL(C_DOUBLE) :: R0 = 320.0_C_DOUBLE
!       LOGICAL :: collision_interspecies = .TRUE.
!       LOGICAL :: SOL = .FALSE.
!       LOGICAL :: LIM = .FALSE.
!       INTEGER :: ir_SOL_separatrix = 32
!       REAL(C_DOUBLE) :: dt = 50.0_C_DOUBLE
!       REAL(C_DOUBLE) :: dvpar = 0.18897637795_C_DOUBLE

!       REAL(C_DOUBLE), DIMENSION(:,:), ALLOCATABLE :: comb_mat
!       REAL(C_DOUBLE), DIMENSION(:), ALLOCATABLE :: hat_As
!       REAL(C_DOUBLE), DIMENSION(:), ALLOCATABLE :: hat_Zs
!       REAL(C_DOUBLE), DIMENSION(:), ALLOCATABLE :: mug
!       REAL(C_DOUBLE), DIMENSION(:), ALLOCATABLE :: vparg
!       REAL(C_DOUBLE), DIMENSION(:), ALLOCATABLE :: coeff_intdmu
!       REAL(C_DOUBLE), DIMENSION(:), ALLOCATABLE :: coeff_intdvpar
!       REAL(C_DOUBLE), DIMENSION(:), ALLOCATABLE :: rg
!       REAL(C_DOUBLE), DIMENSION(:), ALLOCATABLE :: nustar0_r
!       REAL(C_DOUBLE), DIMENSION(:), ALLOCATABLE :: q_r
!       REAL(C_DOUBLE), DIMENSION(:), ALLOCATABLE :: mask_buffer_r
!       REAL(C_DOUBLE), DIMENSION(:,:), ALLOCATABLE :: mask_LIM
!       REAL(C_DOUBLE), DIMENSION(:,:), ALLOCATABLE :: B_norm
!       REAL(C_DOUBLE), DIMENSION(:,:,:,:), ALLOCATABLE :: Bstar_s

!       INTEGER(koliop_Status) :: an_error_code

!       REAL(C_DOUBLE), DIMENSION(:,:,:,:,:,:), ALLOCATABLE :: f5d_s

!       TYPE(C_PTR) :: an_operator_handle

!       allocate(comb_mat(index_max + 1 + 1, index_max + 1 + 1))
!       allocate(hat_As(the_species_extent))
!       allocate(hat_Zs(the_species_extent))
!       allocate(mug(the_mu_extent))
!       allocate(vparg(the_vpar_extent))
!       allocate(coeff_intdmu(the_mu_extent))
!       allocate(coeff_intdvpar(the_vpar_extent))
!       allocate(rg(the_r_extent))
!       allocate(nustar0_r(the_r_extent))
!       allocate(q_r(the_r_extent))
!       allocate(mask_buffer_r(the_r_extent))
!       allocate(mask_LIM(the_r_extent, the_theta_extent))
!       allocate(B_norm(the_r_extent, the_theta_extent + 1)) ! This +1 is weird.
!       allocate(Bstar_s(the_vpar_extent, the_r_extent, the_theta_extent, the_species_extent))
!       allocate(f5d_s(the_mu_extent, the_vpar_extent, the_r_extent, the_theta_extent, the_phi_extent, the_species_extent))

!       comb_mat = 42.0_C_DOUBLE
!       hat_As = 42.0_C_DOUBLE
!       hat_Zs = 42.0_C_DOUBLE
!       mug = 42.0_C_DOUBLE
!       vparg = 42.0_C_DOUBLE
!       coeff_intdmu = 42.0_C_DOUBLE
!       coeff_intdvpar = 42.0_C_DOUBLE
!       rg = 42.0_C_DOUBLE
!       nustar0_r = 42.0_C_DOUBLE
!       q_r = 42.0_C_DOUBLE
!       mask_buffer_r = 42.0_C_DOUBLE
!       mask_LIM = 42.0_C_DOUBLE
!       B_norm = 42.0_C_DOUBLE
!       Bstar_s = 42.0_C_DOUBLE
!       f5d_s = 42.0_C_DOUBLE

!       call koliop_Initialize_f(a_host_thread_count, &
!          a_device_index, &
!          with_verbose_initialization, &
!          an_error_code)
!       print *, "koliop_Initialize_f:an_error_code", an_error_code

!       call koliop_Create_f(an_operator_handle, &
!          Npolmax, index_max, &
!          pSpecies_id, mu_id, &
!          the_mu_extent, the_vpar_extent, the_r_extent, &
!          the_theta_extent, the_phi_extent, the_species_extent, &
!          R0,&
!          collision_interspecies, SOL, LIM, &
!          ir_SOL_separatrix, &
!          dt, dvpar, &
!          comb_mat, &
!          hat_As, hat_Zs, &
!          mug, vparg, &
!          coeff_intdmu, coeff_intdvpar, &
!          rg, nustar0_r, q_r, &
!          mask_buffer_r, mask_LIM, &
!          B_norm, Bstar_s, &
!          an_error_code)
!       print *, "koliop_Create_f:an_error_code", an_error_code

!       call koliop_ComputeMoments_f(an_operator_handle, &
!          f5d_s, &
!          an_error_code)
!       print *, "koliop_ComputeMoments_f:an_error_code", an_error_code

!       call koliop_PrecomputeNu_f(an_operator_handle, &
!          an_error_code)
!       print *, "koliop_PrecomputeNu_f:an_error_code", an_error_code

!       call koliop_ComputeMaxwellianEvolution_f(an_operator_handle, &
!          f5d_s, &
!          an_error_code)
!       print *, "koliop_ComputeMaxwellianEvolution_f:an_error_code", an_error_code

!       call koliop_ComputeQparOnNT_f(an_operator_handle, &
!          f5d_s, &
!          an_error_code)
!       print *, "koliop_ComputeQparOnNT_f:an_error_code", an_error_code

!       call koliop_ComputeCpar_f(an_operator_handle, &
!          f5d_s, &
!          an_error_code)
!       print *, "koliop_ComputeCpar_f:an_error_code", an_error_code

!       call koliop_ComputeMaxwellianUnshiftedLoc_f(an_operator_handle, &
!          f5d_s, &
!          4, &
!          an_error_code)
!       print *, "koliop_ComputeMaxwellianUnshiftedLoc_f:an_error_code", an_error_code

!       call koliop_Apply_f(an_operator_handle, &
!          f5d_s, &
!          an_error_code)
!       print *, "koliop_Apply_f:an_error_code", an_error_code

!       call koliop_ComputeDensityCorrection_f(an_operator_handle, &
!          f5d_s, &
!          an_error_code)
!       print *, "koliop_ComputeDensityCorrection_f:an_error_code", an_error_code

!       call koliop_ComputeUparTCorrection_f(an_operator_handle, &
!          f5d_s, &
!          an_error_code)
!       print *, "koliop_ComputeUparTCorrection_f:an_error_code", an_error_code

!       call koliop_Release_f(an_operator_handle, &
!          an_error_code)
!       print *, "koliop_Release_f:an_error_code", an_error_code

!       call koliop_Deinitialize_f(an_error_code)
!       print *, "koliop_Deinitialize_f:an_error_code", an_error_code
!    END SUBROUTINE
! END PROGRAM
