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

   PUBLIC :: KOLIOP_EXTRACTION_IDENTIFIER_nustar_coeff
   PUBLIC :: KOLIOP_EXTRACTION_IDENTIFIER_n_s
   PUBLIC :: KOLIOP_EXTRACTION_IDENTIFIER_U_par_s
   PUBLIC :: KOLIOP_EXTRACTION_IDENTIFIER_T_s
   PUBLIC :: KOLIOP_EXTRACTION_IDENTIFIER_nu_ab
   PUBLIC :: KOLIOP_EXTRACTION_IDENTIFIER_q_par_on_nT_s

   PUBLIC :: koliop_Create_fp
   PUBLIC :: koliop_ComputeMoments_fp
   PUBLIC :: koliop_ComputeMaxwellianEvolution_fp
   PUBLIC :: koliop_ComputeQparOnNT_fp
   PUBLIC :: koliop_ComputeCpar_fp
   PUBLIC :: koliop_ComputeMaxwellianUnshiftedLoc_fp
   PUBLIC :: koliop_Apply_fp
   PUBLIC :: koliop_ComputeDensityCorrection_fp
   PUBLIC :: koliop_ComputeUparTCorrection_fp
   PUBLIC :: koliop_ExtractRaw_fp

   PUBLIC :: koliop_Initialize_f
   PUBLIC :: koliop_Deinitialize_f
   PUBLIC :: koliop_Create_f
   PUBLIC :: koliop_Release_f
   PUBLIC :: koliop_Fence_f
   PUBLIC :: koliop_ComputeMoments_f
   PUBLIC :: koliop_PrecomputeNu_f
   PUBLIC :: koliop_ComputeMaxwellianEvolution_f
   PUBLIC :: koliop_ComputeQparOnNT_f
   PUBLIC :: koliop_ComputeCpar_f
   PUBLIC :: koliop_ComputeMaxwellianUnshiftedLoc_f
   PUBLIC :: koliop_Apply_f
   PUBLIC :: koliop_ComputeDensityCorrection_f
   PUBLIC :: koliop_ComputeUparTCorrection_f
   PUBLIC :: koliop_ExtractRaw_f

   INTEGER, PARAMETER :: koliop_Status = C_INT32_T

   INTEGER(koliop_Status), PARAMETER :: KOLIOP_STATUS_SUCCESS = 0
   INTEGER(koliop_Status), PARAMETER :: KOLIOP_STATUS_UNKOWN_FAILURE = -1
   INTEGER(koliop_Status), PARAMETER :: KOLIOP_STATUS_INITIALIZATION_FAILURE = -2
   INTEGER(koliop_Status), PARAMETER :: KOLIOP_STATUS_DEINITIALIZATION_FAILURE = -3
   INTEGER(koliop_Status), PARAMETER :: KOLIOP_STATUS_INVALID_VALUE = -30

   INTEGER(C_INT32_T), PARAMETER :: KOLIOP_EXTRACTION_IDENTIFIER_nustar_coeff = 1
   INTEGER(C_INT32_T), PARAMETER :: KOLIOP_EXTRACTION_IDENTIFIER_n_s = 2
   INTEGER(C_INT32_T), PARAMETER :: KOLIOP_EXTRACTION_IDENTIFIER_U_par_s = 3
   INTEGER(C_INT32_T), PARAMETER :: KOLIOP_EXTRACTION_IDENTIFIER_T_s = 4
   INTEGER(C_INT32_T), PARAMETER :: KOLIOP_EXTRACTION_IDENTIFIER_nu_ab = 5
   INTEGER(C_INT32_T), PARAMETER :: KOLIOP_EXTRACTION_IDENTIFIER_q_par_on_nT_s = 6

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
      ! koliop_Status koliop_Initialize(int8_t do_kokkos_initialization,
      !                                 size_t a_host_thread_count,
      !                                 size_t a_device_index,
      !                                 int8_t with_verbose_initialization);
      INTEGER(koliop_Status) FUNCTION koliop_Initialize_c( &
         do_kokkos_initialization, &
         a_host_thread_count, &
         a_device_index, &
         with_verbose_initialization) &
         BIND(C, NAME='koliop_Initialize')

         IMPORT :: koliop_Status
         IMPORT :: C_INT8_T
         IMPORT :: C_SIZE_T

         IMPLICIT NONE

         INTEGER(C_INT8_T), VALUE, INTENT(IN) :: do_kokkos_initialization
         INTEGER(C_SIZE_T), VALUE, INTENT(IN) :: a_host_thread_count
         INTEGER(C_SIZE_T), VALUE, INTENT(IN) :: a_device_index
         INTEGER(C_INT8_T), VALUE, INTENT(IN) :: with_verbose_initialization
      END FUNCTION

      ! koliop_Status koliop_Deinitialize(void);
      INTEGER(koliop_Status) FUNCTION koliop_Deinitialize_c( &
         do_kokkos_deinitialization) &
         BIND(C, NAME='koliop_Deinitialize')

         IMPORT :: koliop_Status
         IMPORT :: C_INT8_T

         IMPLICIT NONE

         INTEGER(C_INT8_T), VALUE, INTENT(IN) :: do_kokkos_deinitialization
      END FUNCTION

      ! koliop_Status koliop_Create(koliop_Operator* an_operator_handle,
      !                             size_t           mu_extent,
      !                             size_t           vpar_extent,
      !                             size_t           tor1_extent,
      !                             size_t           tor2_extent,
      !                             size_t           tor3_extent,
      !                             size_t           species_extent,
      !                             size_t           global_domain_r_extent,
      !                             size_t           global_domain_theta_extent,
      !                             size_t           local_domain_r_offset,
      !                             size_t           local_domain_theta_offset,
      !                             int8_t           collision_interspecies,
      !                             int8_t           SOL,
      !                             int8_t           LIM,
      !                             int64_t          ir_SOL_separatrix,
      !                             const double*    hat_As,
      !                             const double*    hat_Zs,
      !                             const double*    mug,
      !                             const double*    vparg,
      !                             const double*    coeff_intdmu,
      !                             const double*    coeff_intdvpar,
      !                             const double*    coeff_AD_r,
      !                             const double*    mask_buffer_r,
      !                             const double*    mask_LIM,
      !                             const double*    B_norm,
      !                             const double*    Bstar_s);
      INTEGER(koliop_Status) FUNCTION koliop_Create_c( &
         an_operator_handle, &
         mu_extent, &
         vpar_extent, &
         tor1_extent, &
         tor2_extent, &
         tor3_extent, &
         species_extent, &
         global_domain_r_extent, &
         global_domain_theta_extent, &
         local_domain_r_offset, &
         local_domain_theta_offset, &
         collision_interspecies, &
         SOL, &
         LIM, &
         ir_SOL_separatrix, &
         hat_As, &
         hat_Zs, &
         mug, &
         vparg, &
         coeff_intdmu, &
         coeff_intdvpar, &
         coeff_AD_r, &
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

         ! NOTE: the C code expects a pointer to a pointer for
         ! an_operator_handle
         TYPE(C_PTR), INTENT(INOUT) :: an_operator_handle
         INTEGER(C_SIZE_T), VALUE, INTENT(IN) :: mu_extent
         INTEGER(C_SIZE_T), VALUE, INTENT(IN) :: vpar_extent
         INTEGER(C_SIZE_T), VALUE, INTENT(IN) :: tor1_extent
         INTEGER(C_SIZE_T), VALUE, INTENT(IN) :: tor2_extent
         INTEGER(C_SIZE_T), VALUE, INTENT(IN) :: tor3_extent
         INTEGER(C_SIZE_T), VALUE, INTENT(IN) :: species_extent
         INTEGER(C_SIZE_T), VALUE, INTENT(IN) :: global_domain_r_extent
         INTEGER(C_SIZE_T), VALUE, INTENT(IN) :: global_domain_theta_extent
         INTEGER(C_SIZE_T), VALUE, INTENT(IN) :: local_domain_r_offset
         INTEGER(C_SIZE_T), VALUE, INTENT(IN) :: local_domain_theta_offset
         INTEGER(C_INT8_T), VALUE, INTENT(IN) :: collision_interspecies
         INTEGER(C_INT8_T), VALUE, INTENT(IN) :: SOL
         INTEGER(C_INT8_T), VALUE, INTENT(IN) :: LIM
         INTEGER(C_INT64_T), VALUE, INTENT(IN) :: ir_SOL_separatrix
         TYPE(C_PTR), VALUE, INTENT(IN) :: hat_As
         TYPE(C_PTR), VALUE, INTENT(IN) :: hat_Zs
         TYPE(C_PTR), VALUE, INTENT(IN) :: mug
         TYPE(C_PTR), VALUE, INTENT(IN) :: vparg
         TYPE(C_PTR), VALUE, INTENT(IN) :: coeff_intdmu
         TYPE(C_PTR), VALUE, INTENT(IN) :: coeff_intdvpar
         TYPE(C_PTR), VALUE, INTENT(IN) :: coeff_AD_r
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

      ! koliop_Status koliop_Fence(koliop_Operator an_operator)
      INTEGER(koliop_Status) FUNCTION koliop_Fence_c( &
         an_operator) &
         BIND(C, NAME='koliop_Fence')

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
         dt, &
         f5d_s) &
         BIND(C, NAME='koliop_ComputeMaxwellianEvolution')

         IMPORT :: koliop_Status
         IMPORT :: C_PTR
         IMPORT :: C_DOUBLE

         IMPLICIT NONE

         TYPE(C_PTR), VALUE, INTENT(IN) :: an_operator
         REAL(C_DOUBLE), VALUE, INTENT(IN) :: dt
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
         dt, &
         f5d_s) &
         BIND(C, NAME='koliop_ComputeCpar')

         IMPORT :: koliop_Status
         IMPORT :: C_PTR
         IMPORT :: C_DOUBLE

         IMPLICIT NONE

         TYPE(C_PTR), VALUE, INTENT(IN) :: an_operator
         REAL(C_DOUBLE), VALUE, INTENT(IN) :: dt
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
         dt, &
         f5d_s) &
         BIND(C, NAME='koliop_Apply')

         IMPORT :: koliop_Status
         IMPORT :: C_PTR
         IMPORT :: C_DOUBLE

         IMPLICIT NONE

         TYPE(C_PTR), VALUE, INTENT(IN) :: an_operator
         REAL(C_DOUBLE), VALUE, INTENT(IN) :: dt
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
      !                                             double          dt,
      !                                             double*         f5d_s);
      INTEGER(koliop_Status) FUNCTION koliop_ComputeUparTCorrection_c( &
         an_operator, &
         dt, &
         f5d_s) &
         BIND(C, NAME='koliop_ComputeUparTCorrection')

         IMPORT :: koliop_Status
         IMPORT :: C_PTR
         IMPORT :: C_DOUBLE

         IMPLICIT NONE

         TYPE(C_PTR), VALUE, INTENT(IN) :: an_operator
         REAL(C_DOUBLE), VALUE, INTENT(IN) :: dt
         TYPE(C_PTR), VALUE, INTENT(IN) :: f5d_s
      END FUNCTION

      ! koliop_Status koliop_ExtractRaw(koliop_Operator an_operator,
      !                                 int32_t         value_identifier,
      !                                 void*           an_output_buffer);
      INTEGER(koliop_Status) FUNCTION koliop_ExtractRaw_c( &
         an_operator, &
         value_identifier, &
         an_output_buffer) &
         BIND(C, NAME='koliop_ExtractRaw')

         IMPORT :: koliop_Status
         IMPORT :: C_PTR
         IMPORT :: C_INT32_T

         IMPLICIT NONE

         TYPE(C_PTR), VALUE, INTENT(IN) :: an_operator
         INTEGER(C_INT32_T), VALUE, INTENT(IN) :: value_identifier
         TYPE(C_PTR), VALUE, INTENT(IN) :: an_output_buffer
      END FUNCTION

      ! koliop_Status koliop_Collision_c(koliop_Operator an_operator,
      !                                  double          dt,
      !                                  double*         f5d_s);
      INTEGER(koliop_Status) FUNCTION koliop_Collision_c( &
         an_operator, &
         dt, &
         f5d_s) &
         BIND(C, NAME='koliop_Collision')

         IMPORT :: koliop_Status
         IMPORT :: C_PTR
         IMPORT :: C_DOUBLE

         IMPLICIT NONE

         TYPE(C_PTR), VALUE, INTENT(IN) :: an_operator
         REAL(C_DOUBLE), VALUE, INTENT(IN) :: dt
         TYPE(C_PTR), VALUE, INTENT(IN) :: f5d_s
      END FUNCTION
   END INTERFACE

CONTAINS

   ! NOTE: Assuming use do some part of the Gysela computation using OpenMP on
   ! GPU, you should use omp_get_mapped_ptr, use_device_ptr etc. to pass data to
   ! Koliop. These OpenMP calls and directives return C_PTR thus the interface
   ! accepts C_PTR and not DIMENSION(*) which is a same for type safety.. Yay
   ! Fortran.
   ! For this reason we offer two version of the interface. Routines suffixed
   ! with "_f" take plain DIMENTION(*) array as input and convert them to C_PTR.
   ! "_fp" suffixed routines take C_PTR as input (for use with OpenMP).

   !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!

   SUBROUTINE koliop_Create_fp(&
      an_operator_handle, &
      mu_extent, &
      vpar_extent, &
      tor1_extent, &
      tor2_extent, &
      tor3_extent, &
      species_extent, &
      global_domain_r_extent, &
      global_domain_theta_extent, &
      local_domain_r_offset, &
      local_domain_theta_offset, &
      collision_interspecies, &
      SOL, &
      LIM, &
      ir_SOL_separatrix, &
      hat_As, &
      hat_Zs, &
      mug, &
      vparg, &
      coeff_intdmu, &
      coeff_intdvpar, &
      coeff_AD_r, &
      mask_buffer_r, &
      mask_LIM, &
      B_norm, &
      Bstar_s, &
      an_error_code)

      IMPLICIT NONE

      TYPE(C_PTR), INTENT(INOUT) :: an_operator_handle
      INTEGER, INTENT(IN) :: mu_extent
      INTEGER, INTENT(IN) :: vpar_extent
      INTEGER, INTENT(IN) :: tor1_extent
      INTEGER, INTENT(IN) :: tor2_extent
      INTEGER, INTENT(IN) :: tor3_extent
      INTEGER, INTENT(IN) :: species_extent
      INTEGER, INTENT(IN) :: global_domain_r_extent
      INTEGER, INTENT(IN) :: global_domain_theta_extent
      INTEGER, INTENT(IN) :: local_domain_r_offset
      INTEGER, INTENT(IN) :: local_domain_theta_offset
      LOGICAL, INTENT(IN) :: collision_interspecies
      LOGICAL, INTENT(IN) :: SOL
      LOGICAL, INTENT(IN) :: LIM
      INTEGER, INTENT(IN) :: ir_SOL_separatrix
      ! REAL(C_DOUBLE), TARGET, DIMENSION(*), INTENT(IN) :: hat_As
      TYPE(C_PTR), INTENT(IN) :: hat_As
      ! REAL(C_DOUBLE), TARGET, DIMENSION(*), INTENT(IN) :: hat_Zs
      TYPE(C_PTR), INTENT(IN) :: hat_Zs
      ! REAL(C_DOUBLE), TARGET, DIMENSION(*), INTENT(IN) :: mug
      TYPE(C_PTR), INTENT(IN) :: mug
      ! REAL(C_DOUBLE), TARGET, DIMENSION(*), INTENT(IN) :: vparg
      TYPE(C_PTR), INTENT(IN) :: vparg
      ! REAL(C_DOUBLE), TARGET, DIMENSION(*), INTENT(IN) :: coeff_intdmu
      TYPE(C_PTR), INTENT(IN) :: coeff_intdmu
      ! REAL(C_DOUBLE), TARGET, DIMENSION(*), INTENT(IN) :: coeff_intdvpar
      TYPE(C_PTR), INTENT(IN) :: coeff_intdvpar
      ! REAL(C_DOUBLE), TARGET, DIMENSION(*), INTENT(IN) :: coeff_AD_r
      TYPE(C_PTR), INTENT(IN) :: coeff_AD_r
      ! REAL(C_DOUBLE), TARGET, DIMENSION(*), INTENT(IN) :: mask_buffer_r
      TYPE(C_PTR), INTENT(IN) :: mask_buffer_r
      ! REAL(C_DOUBLE), TARGET, DIMENSION(*), INTENT(IN) :: mask_LIM
      TYPE(C_PTR), INTENT(IN) :: mask_LIM
      ! REAL(C_DOUBLE), TARGET, DIMENSION(*), INTENT(IN) :: B_norm
      TYPE(C_PTR), INTENT(IN) :: B_norm
      ! REAL(C_DOUBLE), TARGET, DIMENSION(*), INTENT(IN) :: Bstar_s
      TYPE(C_PTR), INTENT(IN) :: Bstar_s
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
         INT(mu_extent, C_SIZE_T), &
         INT(vpar_extent, C_SIZE_T), &
         INT(tor1_extent, C_SIZE_T), &
         INT(tor2_extent, C_SIZE_T), &
         INT(tor3_extent, C_SIZE_T), &
         INT(species_extent, C_SIZE_T), &
         INT(global_domain_r_extent, C_SIZE_T), &
         INT(global_domain_theta_extent, C_SIZE_T), &
         INT(local_domain_r_offset, C_SIZE_T), &
         INT(local_domain_theta_offset, C_SIZE_T), &
         collision_interspecies_cast, &
         SOL_cast, &
         LIM_cast, &
         INT(ir_SOL_separatrix, C_INT64_T), &
         hat_As, &
         hat_Zs, &
         mug, &
         vparg, &
         coeff_intdmu, &
         coeff_intdvpar, &
         coeff_AD_r, &
         mask_buffer_r, &
         mask_LIM, &
         B_norm, &
         Bstar_s)
   END SUBROUTINE

   SUBROUTINE koliop_ComputeMoments_fp( &
      an_operator_handle, &
      f5d_s, &
      an_error_code)

      IMPLICIT NONE

      TYPE(C_PTR), INTENT(IN) :: an_operator_handle
      ! REAL(C_DOUBLE), TARGET, DIMENSION(*), INTENT(INOUT) :: f5d_s
      TYPE(C_PTR), INTENT(IN) :: f5d_s
      INTEGER(koliop_Status), INTENT(OUT) :: an_error_code

      an_error_code = koliop_ComputeMoments_c( &
         an_operator_handle, &
         f5d_s)
   END SUBROUTINE

   SUBROUTINE koliop_ComputeMaxwellianEvolution_fp( &
      an_operator_handle, &
      dt, &
      f5d_s, &
      an_error_code)

      IMPLICIT NONE

      TYPE(C_PTR), INTENT(IN) :: an_operator_handle
      REAL(C_DOUBLE), INTENT(IN) :: dt
      ! REAL(C_DOUBLE), TARGET, DIMENSION(*), INTENT(INOUT) :: f5d_s
      TYPE(C_PTR), INTENT(IN) :: f5d_s
      INTEGER(koliop_Status), INTENT(OUT) :: an_error_code

      an_error_code = koliop_ComputeMaxwellianEvolution_c( &
         an_operator_handle, &
         dt, &
         f5d_s)
   END SUBROUTINE

   SUBROUTINE koliop_ComputeQparOnNT_fp( &
      an_operator_handle, &
      f5d_s, &
      an_error_code)

      IMPLICIT NONE

      TYPE(C_PTR), INTENT(IN) :: an_operator_handle
      ! REAL(C_DOUBLE), TARGET, DIMENSION(*), INTENT(IN) :: f5d_s
      TYPE(C_PTR), INTENT(IN) :: f5d_s
      INTEGER(koliop_Status), INTENT(OUT) :: an_error_code

      an_error_code = koliop_ComputeQparOnNT_c( &
         an_operator_handle, &
         f5d_s)
   END SUBROUTINE

   SUBROUTINE koliop_ComputeCpar_fp( &
      an_operator_handle, &
      dt, &
      f5d_s, &
      an_error_code)

      IMPLICIT NONE

      TYPE(C_PTR), INTENT(IN) :: an_operator_handle
      REAL(C_DOUBLE), INTENT(IN) :: dt
      ! REAL(C_DOUBLE), TARGET, DIMENSION(*), INTENT(INOUT) :: f5d_s
      TYPE(C_PTR), INTENT(IN) :: f5d_s
      INTEGER(koliop_Status), INTENT(OUT) :: an_error_code

      an_error_code = koliop_ComputeCpar_c( &
         an_operator_handle, &
         dt, &
         f5d_s)
   END SUBROUTINE

   SUBROUTINE koliop_ComputeMaxwellianUnshiftedLoc_fp( &
      an_operator_handle, &
      f5d_s, &
      choice, &
      an_error_code)

      IMPLICIT NONE

      TYPE(C_PTR), INTENT(IN) :: an_operator_handle
      ! REAL(C_DOUBLE), TARGET, DIMENSION(*), INTENT(INOUT) :: f5d_s
      TYPE(C_PTR), INTENT(IN) :: f5d_s
      INTEGER, INTENT(IN) :: choice
      INTEGER(koliop_Status), INTENT(OUT) :: an_error_code

      an_error_code = koliop_ComputeMaxwellianUnshiftedLoc_c( &
         an_operator_handle, &
         f5d_s, &
         INT(choice, C_INT32_T))
   END SUBROUTINE

   SUBROUTINE koliop_Apply_fp( &
      an_operator_handle, &
      dt, &
      f5d_s, &
      an_error_code)

      IMPLICIT NONE

      TYPE(C_PTR), INTENT(IN) :: an_operator_handle
      REAL(C_DOUBLE), INTENT(IN) :: dt
      ! REAL(C_DOUBLE), TARGET, DIMENSION(*), INTENT(INOUT) :: f5d_s
      TYPE(C_PTR), INTENT(IN) :: f5d_s
      INTEGER(koliop_Status), INTENT(OUT) :: an_error_code

      an_error_code = koliop_Apply_c( &
         an_operator_handle, &
         dt, &
         f5d_s)
   END SUBROUTINE

   SUBROUTINE koliop_ComputeDensityCorrection_fp( &
      an_operator_handle, &
      f5d_s, &
      an_error_code)

      IMPLICIT NONE

      TYPE(C_PTR), INTENT(IN) :: an_operator_handle
      ! REAL(C_DOUBLE), TARGET, DIMENSION(*), INTENT(INOUT) :: f5d_s
      TYPE(C_PTR), INTENT(IN) :: f5d_s
      INTEGER(koliop_Status), INTENT(OUT) :: an_error_code

      an_error_code = koliop_ComputeDensityCorrection_c( &
         an_operator_handle, &
         f5d_s)
   END SUBROUTINE

   SUBROUTINE koliop_ComputeUparTCorrection_fp( &
      an_operator_handle, &
      dt, &
      f5d_s, &
      an_error_code)

      IMPLICIT NONE

      TYPE(C_PTR), INTENT(IN) :: an_operator_handle
      REAL(C_DOUBLE), INTENT(IN) :: dt
      ! REAL(C_DOUBLE), TARGET, DIMENSION(*), INTENT(INOUT) :: f5d_s
      TYPE(C_PTR), INTENT(IN) :: f5d_s
      INTEGER(koliop_Status), INTENT(OUT) :: an_error_code

      an_error_code = koliop_ComputeUparTCorrection_c( &
         an_operator_handle, &
         dt, &
         f5d_s)
   END SUBROUTINE

   SUBROUTINE koliop_ExtractRaw_fp( &
      an_operator_handle, &
      value_identifier, &
      an_output_buffer, &
      an_error_code)

      IMPLICIT NONE

      TYPE(C_PTR), INTENT(IN) :: an_operator_handle
      INTEGER, INTENT(IN) :: value_identifier
      ! REAL(C_DOUBLE), TARGET, DIMENSION(*), INTENT(INOUT) :: an_output_buffer
      TYPE(C_PTR), INTENT(IN) :: an_output_buffer
      INTEGER(koliop_Status), INTENT(OUT) :: an_error_code

      an_error_code = koliop_ExtractRaw_c( &
         an_operator_handle, &
         INT(value_identifier, C_INT32_T), &
         an_output_buffer)
   END SUBROUTINE

   SUBROUTINE koliop_Collision_fp( &
      an_operator_handle, &
      dt, &
      f5d_s, &
      an_error_code)

      IMPLICIT NONE

      TYPE(C_PTR), INTENT(IN) :: an_operator_handle
      REAL(C_DOUBLE), INTENT(IN) :: dt
      ! REAL(C_DOUBLE), TARGET, DIMENSION(*), INTENT(INOUT) :: f5d_s
      TYPE(C_PTR), INTENT(IN) :: f5d_s
      INTEGER(koliop_Status), INTENT(OUT) :: an_error_code

      an_error_code = koliop_Collision_c( &
         an_operator_handle, &
         dt, &
         f5d_s)
   END SUBROUTINE

!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!

   SUBROUTINE koliop_Initialize_f( &
      do_kokkos_initialization, &
      a_host_thread_count, &
      a_device_index, &
      with_verbose_initialization, &
      an_error_code)

      IMPLICIT NONE

      LOGICAL, INTENT(IN) :: do_kokkos_initialization
      INTEGER, INTENT(IN) :: a_host_thread_count
      INTEGER, INTENT(IN) :: a_device_index
      LOGICAL, INTENT(IN) :: with_verbose_initialization
      INTEGER(koliop_Status), INTENT(OUT) :: an_error_code

      INTEGER(C_INT8_T) :: do_kokkos_initialization_cast
      INTEGER(C_INT8_T) :: with_verbose_initialization_cast

      IF (do_kokkos_initialization) THEN
         do_kokkos_initialization_cast = 1
      ELSE
         do_kokkos_initialization_cast = 0
      END IF

      IF (with_verbose_initialization) THEN
         with_verbose_initialization_cast = 1
      ELSE
         with_verbose_initialization_cast = 0
      END IF

      an_error_code = koliop_Initialize_c( &
         do_kokkos_initialization_cast, &
         INT(a_host_thread_count, C_SIZE_T), &
         INT(a_device_index, C_SIZE_T), &
         with_verbose_initialization_cast)
   END SUBROUTINE

   SUBROUTINE koliop_Deinitialize_f( &
      do_kokkos_deinitialization, &
      an_error_code)

      IMPLICIT NONE

      LOGICAL, INTENT(IN) :: do_kokkos_deinitialization
      INTEGER(koliop_Status), INTENT(OUT) :: an_error_code

      INTEGER(C_INT8_T) :: do_kokkos_deinitialization_cast

      IF (do_kokkos_deinitialization) THEN
         do_kokkos_deinitialization_cast = 1
      ELSE
         do_kokkos_deinitialization_cast = 0
      END IF

      an_error_code = koliop_Deinitialize_c(do_kokkos_deinitialization_cast)
   END SUBROUTINE

   SUBROUTINE koliop_Create_f(&
      an_operator_handle, &
      mu_extent, &
      vpar_extent, &
      tor1_extent, &
      tor2_extent, &
      tor3_extent, &
      species_extent, &
      global_domain_r_extent, &
      global_domain_theta_extent, &
      local_domain_r_offset, &
      local_domain_theta_offset, &
      collision_interspecies, &
      SOL, &
      LIM, &
      ir_SOL_separatrix, &
      hat_As, &
      hat_Zs, &
      mug, &
      vparg, &
      coeff_intdmu, &
      coeff_intdvpar, &
      coeff_AD_r, &
      mask_buffer_r, &
      mask_LIM, &
      B_norm, &
      Bstar_s, &
      an_error_code)

      IMPLICIT NONE

      TYPE(C_PTR), INTENT(INOUT) :: an_operator_handle
      INTEGER, INTENT(IN) :: mu_extent
      INTEGER, INTENT(IN) :: vpar_extent
      INTEGER, INTENT(IN) :: tor1_extent
      INTEGER, INTENT(IN) :: tor2_extent
      INTEGER, INTENT(IN) :: tor3_extent
      INTEGER, INTENT(IN) :: species_extent
      INTEGER, INTENT(IN) :: global_domain_r_extent
      INTEGER, INTENT(IN) :: global_domain_theta_extent
      INTEGER, INTENT(IN) :: local_domain_r_offset
      INTEGER, INTENT(IN) :: local_domain_theta_offset
      LOGICAL, INTENT(IN) :: collision_interspecies
      LOGICAL, INTENT(IN) :: SOL
      LOGICAL, INTENT(IN) :: LIM
      INTEGER, INTENT(IN) :: ir_SOL_separatrix
      REAL(C_DOUBLE), TARGET, DIMENSION(*), INTENT(IN) :: hat_As
      REAL(C_DOUBLE), TARGET, DIMENSION(*), INTENT(IN) :: hat_Zs
      REAL(C_DOUBLE), TARGET, DIMENSION(*), INTENT(IN) :: mug
      REAL(C_DOUBLE), TARGET, DIMENSION(*), INTENT(IN) :: vparg
      REAL(C_DOUBLE), TARGET, DIMENSION(*), INTENT(IN) :: coeff_intdmu
      REAL(C_DOUBLE), TARGET, DIMENSION(*), INTENT(IN) :: coeff_intdvpar
      REAL(C_DOUBLE), TARGET, DIMENSION(*), INTENT(IN) :: coeff_AD_r
      REAL(C_DOUBLE), TARGET, DIMENSION(*), INTENT(IN) :: mask_buffer_r
      REAL(C_DOUBLE), TARGET, DIMENSION(*), INTENT(IN) :: mask_LIM
      REAL(C_DOUBLE), TARGET, DIMENSION(*), INTENT(IN) :: B_norm
      REAL(C_DOUBLE), TARGET, DIMENSION(*), INTENT(IN) :: Bstar_s
      INTEGER(koliop_Status), INTENT(OUT) :: an_error_code

      call koliop_Create_fp( &
         an_operator_handle, &
         mu_extent, &
         vpar_extent, &
         tor1_extent, &
         tor2_extent, &
         tor3_extent, &
         species_extent, &
         global_domain_r_extent, &
         global_domain_theta_extent, &
         local_domain_r_offset, &
         local_domain_theta_offset, &
         collision_interspecies, &
         SOL, &
         LIM, &
         ir_SOL_separatrix, &
         C_LOC(hat_As), &
         C_LOC(hat_Zs), &
         C_LOC(mug), &
         C_LOC(vparg), &
         C_LOC(coeff_intdmu), &
         C_LOC(coeff_intdvpar), &
         C_LOC(coeff_AD_r), &
         C_LOC(mask_buffer_r), &
         C_LOC(mask_LIM), &
         C_LOC(B_norm), &
         C_LOC(Bstar_s), &
         an_error_code)
   END SUBROUTINE

   SUBROUTINE koliop_Release_f( &
      an_operator_handle, &
      an_error_code)

      IMPLICIT NONE

      TYPE(C_PTR), INTENT(IN) :: an_operator_handle
      INTEGER(koliop_Status), INTENT(OUT) :: an_error_code

      an_error_code = koliop_Release_c(an_operator_handle)
   END SUBROUTINE

   SUBROUTINE koliop_Fence_f( &
      an_operator_handle, &
      an_error_code)

      IMPLICIT NONE

      TYPE(C_PTR), INTENT(IN) :: an_operator_handle
      INTEGER(koliop_Status), INTENT(OUT) :: an_error_code

      an_error_code = koliop_Fence_c(an_operator_handle)
   END SUBROUTINE

   SUBROUTINE koliop_ComputeMoments_f( &
      an_operator_handle, &
      f5d_s, &
      an_error_code)

      IMPLICIT NONE

      TYPE(C_PTR), INTENT(IN) :: an_operator_handle
      REAL(C_DOUBLE), TARGET, DIMENSION(*), INTENT(INOUT) :: f5d_s
      INTEGER(koliop_Status), INTENT(OUT) :: an_error_code

      call koliop_ComputeMoments_fp( &
         an_operator_handle, &
         C_LOC(f5d_s), &
         an_error_code)
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
      dt, &
      f5d_s, &
      an_error_code)

      IMPLICIT NONE

      TYPE(C_PTR), INTENT(IN) :: an_operator_handle
      REAL(C_DOUBLE), INTENT(IN) :: dt
      REAL(C_DOUBLE), TARGET, DIMENSION(*), INTENT(INOUT) :: f5d_s
      INTEGER(koliop_Status), INTENT(OUT) :: an_error_code

      call koliop_ComputeMaxwellianEvolution_fp( &
         an_operator_handle, &
         dt, &
         C_LOC(f5d_s), &
         an_error_code)
   END SUBROUTINE

   SUBROUTINE koliop_ComputeQparOnNT_f( &
      an_operator_handle, &
      f5d_s, &
      an_error_code)

      IMPLICIT NONE

      TYPE(C_PTR), INTENT(IN) :: an_operator_handle
      REAL(C_DOUBLE), TARGET, DIMENSION(*), INTENT(IN) :: f5d_s
      INTEGER(koliop_Status), INTENT(OUT) :: an_error_code

      call koliop_ComputeQparOnNT_fp( &
         an_operator_handle, &
         C_LOC(f5d_s), &
         an_error_code)
   END SUBROUTINE

   SUBROUTINE koliop_ComputeCpar_f( &
      an_operator_handle, &
      dt, &
      f5d_s, &
      an_error_code)

      IMPLICIT NONE

      TYPE(C_PTR), INTENT(IN) :: an_operator_handle
      REAL(C_DOUBLE), INTENT(IN) :: dt
      REAL(C_DOUBLE), TARGET, DIMENSION(*), INTENT(INOUT) :: f5d_s
      INTEGER(koliop_Status), INTENT(OUT) :: an_error_code

      call koliop_ComputeCpar_fp( &
         an_operator_handle, &
         dt, &
         C_LOC(f5d_s), &
         an_error_code)
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

      call koliop_ComputeMaxwellianUnshiftedLoc_fp( &
         an_operator_handle, &
         C_LOC(f5d_s), &
         choice, &
         an_error_code)
   END SUBROUTINE

   SUBROUTINE koliop_Apply_f( &
      an_operator_handle, &
      dt, &
      f5d_s, &
      an_error_code)

      IMPLICIT NONE

      TYPE(C_PTR), INTENT(IN) :: an_operator_handle
      REAL(C_DOUBLE), INTENT(IN) :: dt
      REAL(C_DOUBLE), TARGET, DIMENSION(*), INTENT(INOUT) :: f5d_s
      INTEGER(koliop_Status), INTENT(OUT) :: an_error_code

      call koliop_Apply_fp( &
         an_operator_handle, &
         dt, &
         C_LOC(f5d_s), &
         an_error_code)
   END SUBROUTINE

   SUBROUTINE koliop_ComputeDensityCorrection_f( &
      an_operator_handle, &
      f5d_s, &
      an_error_code)

      IMPLICIT NONE

      TYPE(C_PTR), INTENT(IN) :: an_operator_handle
      REAL(C_DOUBLE), TARGET, DIMENSION(*), INTENT(INOUT) :: f5d_s
      INTEGER(koliop_Status), INTENT(OUT) :: an_error_code

      call koliop_ComputeDensityCorrection_fp( &
         an_operator_handle, &
         C_LOC(f5d_s), &
         an_error_code)
   END SUBROUTINE

   SUBROUTINE koliop_ComputeUparTCorrection_f( &
      an_operator_handle, &
      dt, &
      f5d_s, &
      an_error_code)

      IMPLICIT NONE

      TYPE(C_PTR), INTENT(IN) :: an_operator_handle
      REAL(C_DOUBLE), INTENT(IN) :: dt
      REAL(C_DOUBLE), TARGET, DIMENSION(*), INTENT(INOUT) :: f5d_s
      INTEGER(koliop_Status), INTENT(OUT) :: an_error_code

      call koliop_ComputeUparTCorrection_fp( &
         an_operator_handle, &
         dt, &
         C_LOC(f5d_s), &
         an_error_code)
   END SUBROUTINE

   SUBROUTINE koliop_ExtractRaw_f( &
      an_operator_handle, &
      value_identifier, &
      an_output_buffer, &
      an_error_code)

      IMPLICIT NONE

      TYPE(C_PTR), INTENT(IN) :: an_operator_handle
      INTEGER, INTENT(IN) :: value_identifier
      REAL(C_DOUBLE), TARGET, DIMENSION(*), INTENT(INOUT) :: an_output_buffer
      INTEGER(koliop_Status), INTENT(OUT) :: an_error_code

      call koliop_ExtractRaw_fp( &
         an_operator_handle, &
         value_identifier, &
         C_LOC(an_output_buffer), &
         an_error_code)
   END SUBROUTINE

   SUBROUTINE koliop_Collision_f( &
      an_operator_handle, &
      dt, &
      f5d_s, &
      an_error_code)

      IMPLICIT NONE

      TYPE(C_PTR), INTENT(IN) :: an_operator_handle
      REAL(C_DOUBLE), INTENT(IN) :: dt
      REAL(C_DOUBLE), TARGET, DIMENSION(*), INTENT(INOUT) :: f5d_s
      INTEGER(koliop_Status), INTENT(OUT) :: an_error_code

      call koliop_Collision_fp( &
         an_operator_handle, &
         dt, &
         C_LOC(f5d_s), &
         an_error_code)
   END SUBROUTINE
END MODULE

! To test the interface, uncomment the PROGRAM below and build using something
! like:
! ninja && ftn -v -m 2 -O3 -fopenmp ../include/KOLIOP/koliop.f90 -c && cc -v -o test_dummy_test_f koliop.o -Wl,-rpath,/opt/rocm/lib libkoliop.a  _deps/kokkoskernels-build/libkokkoskernels.a _deps/kokkos-build/containers/src/libkokkoscontainers.a _deps/kokkos-build/core/src/libkokkoscore.a -ldl /opt/rocm/lib/libhsa-runtime64.so /opt/rocm/lib/libamdhip64.so _deps/kokkos-build/simd/src/libkokkossimd.a && echo OK

! PROGRAM Testing
!    USE koliop
!    USE ISO_C_Binding

!    IMPLICIT NONE

!    call koliop_test_Run()

! CONTAINS

!    SUBROUTINE koliop_test_Run()

!       IMPLICIT NONE

!       LOGICAL :: do_kokkos_initialization = .TRUE.
!       INTEGER :: a_host_thread_count = 8
!       INTEGER :: a_device_index = 0
!       LOGICAL :: with_verbose_initialization = .TRUE.

!       INTEGER :: mu_extent = 48
!       INTEGER :: vpar_extent = 128
!       INTEGER :: r_extent = 32
!       INTEGER :: theta_extent = 32
!       INTEGER :: phi_extent = 1
!       INTEGER :: global_domain_r_extent = 32
!       INTEGER :: global_domain_theta_extent = 33
!       INTEGER :: local_domain_r_offset = 0
!       INTEGER :: local_domain_theta_offset = 0
!       INTEGER :: species_extent = 1
!       LOGICAL :: collision_interspecies = .TRUE.
!       LOGICAL :: SOL = .FALSE.
!       LOGICAL :: LIM = .FALSE.
!       INTEGER :: ir_SOL_separatrix = 32
!       REAL(C_DOUBLE) :: dt = 50.0_C_DOUBLE

!       REAL(C_DOUBLE), DIMENSION(:), ALLOCATABLE :: hat_As
!       REAL(C_DOUBLE), DIMENSION(:), ALLOCATABLE :: hat_Zs
!       REAL(C_DOUBLE), DIMENSION(:), ALLOCATABLE :: mug
!       REAL(C_DOUBLE), DIMENSION(:), ALLOCATABLE :: vparg
!       REAL(C_DOUBLE), DIMENSION(:), ALLOCATABLE :: coeff_intdmu
!       REAL(C_DOUBLE), DIMENSION(:), ALLOCATABLE :: coeff_intdvpar
!       REAL(C_DOUBLE), DIMENSION(:), ALLOCATABLE :: coeff_AD_r
!       REAL(C_DOUBLE), DIMENSION(:), ALLOCATABLE :: mask_buffer_r
!       REAL(C_DOUBLE), DIMENSION(:,:), ALLOCATABLE :: mask_LIM
!       REAL(C_DOUBLE), DIMENSION(:,:), ALLOCATABLE :: B_norm
!       REAL(C_DOUBLE), DIMENSION(:,:,:,:), ALLOCATABLE :: Bstar_s

!       INTEGER(koliop_Status) :: an_error_code

!       REAL(C_DOUBLE), DIMENSION(:,:,:,:,:,:), ALLOCATABLE :: f5d_s

!       TYPE(C_PTR) :: an_operator

!       REAL(C_DOUBLE), DIMENSION(:,:,:,:), ALLOCATABLE :: n_s

!       allocate(hat_As(species_extent))
!       allocate(hat_Zs(species_extent))
!       allocate(mug(mu_extent))
!       allocate(vparg(vpar_extent))
!       allocate(coeff_intdmu(mu_extent))
!       allocate(coeff_intdvpar(vpar_extent))
!       allocate(coeff_AD_r(r_extent))
!       allocate(mask_buffer_r(global_domain_r_extent))
!       ! allocate(mask_LIM(global_domain_r_extent, global_domain_theta_extent)) ! LIM is false
!       allocate(B_norm(global_domain_r_extent, global_domain_theta_extent))
!       allocate(Bstar_s(vpar_extent, r_extent, theta_extent, species_extent))
!       allocate(f5d_s(mu_extent, vpar_extent, r_extent, theta_extent, phi_extent, species_extent))

!       allocate(n_s(r_extent, theta_extent, phi_extent, species_extent))

!       hat_As = 42.0_C_DOUBLE
!       hat_Zs = 42.0_C_DOUBLE
!       mug = 42.0_C_DOUBLE
!       vparg = 42.0_C_DOUBLE
!       coeff_intdmu = 42.0_C_DOUBLE
!       coeff_intdvpar = 42.0_C_DOUBLE
!       coeff_AD_r = 42.0_C_DOUBLE
!       mask_buffer_r = 42.0_C_DOUBLE
!       ! mask_LIM = 42.0_C_DOUBLE ! LIM is false
!       B_norm = 42.0_C_DOUBLE
!       Bstar_s = 42.0_C_DOUBLE
!       f5d_s = 42.0_C_DOUBLE

!       n_s = 42.0_C_DOUBLE

!       call koliop_Initialize_f(do_kokkos_initialization, &
!          a_host_thread_count, &
!          a_device_index, &
!          with_verbose_initialization, &
!          an_error_code)
!       print *, "koliop_Initialize_f:an_error_code", an_error_code

!       call koliop_Create_f(an_operator, &
!          mu_extent, vpar_extent, r_extent, &
!          theta_extent, phi_extent, species_extent, &
!          global_domain_r_extent, global_domain_theta_extent, &
!          local_domain_r_offset, local_domain_theta_offset, &
!          collision_interspecies, SOL, LIM, &
!          ir_SOL_separatrix, &
!          hat_As, hat_Zs, &
!          mug, vparg, &
!          coeff_intdmu, coeff_intdvpar, &
!          coeff_AD_r, mask_buffer_r, mask_LIM, &
!          B_norm, Bstar_s, &
!          an_error_code)
!       print *, "koliop_Create_f:an_error_code", an_error_code

!       call koliop_ComputeMoments_f(an_operator, &
!          f5d_s, &
!          an_error_code)
!       print *, "koliop_ComputeMoments_f:an_error_code", an_error_code

!       call koliop_PrecomputeNu_f(an_operator, &
!          an_error_code)
!       print *, "koliop_PrecomputeNu_f:an_error_code", an_error_code

!       call koliop_ComputeMaxwellianEvolution_f(an_operator, &
!          dt, &
!          f5d_s, &
!          an_error_code)
!       print *, "koliop_ComputeMaxwellianEvolution_f:an_error_code", an_error_code

!       call koliop_ComputeQparOnNT_f(an_operator, &
!          f5d_s, &
!          an_error_code)
!       print *, "koliop_ComputeQparOnNT_f:an_error_code", an_error_code

!       call koliop_ComputeCpar_f(an_operator, &
!          dt, &
!          f5d_s, &
!          an_error_code)
!       print *, "koliop_ComputeCpar_f:an_error_code", an_error_code

!       call koliop_ComputeMaxwellianUnshiftedLoc_f(an_operator, &
!          f5d_s, &
!          4, &
!          an_error_code)
!       print *, "koliop_ComputeMaxwellianUnshiftedLoc_f:an_error_code", an_error_code

!       call koliop_Fence_f(an_operator, &
!          an_error_code)
!       print *, "koliop_Fence_f:an_error_code", an_error_code

!       call koliop_Apply_f(an_operator, &
!          dt, &
!          f5d_s, &
!          an_error_code)
!       print *, "koliop_Apply_f:an_error_code", an_error_code

!       call koliop_ComputeDensityCorrection_f(an_operator, &
!          f5d_s, &
!          an_error_code)
!       print *, "koliop_ComputeDensityCorrection_f:an_error_code", an_error_code

!       call koliop_ComputeUparTCorrection_f(an_operator, &
!          dt, &
!          f5d_s, &
!          an_error_code)
!       print *, "koliop_ComputeUparTCorrection_f:an_error_code", an_error_code

!       call koliop_ExtractRaw_f(an_operator, &
!          KOLIOP_EXTRACTION_IDENTIFIER_n_s, &
!          n_s, &
!          an_error_code)
!       print *, "koliop_ExtractRaw_f:an_error_code", an_error_code

!       call koliop_Fence_f(an_operator, &
!          an_error_code)
!       print *, "koliop_Fence_f:an_error_code", an_error_code

!       call koliop_Release_f(an_operator, &
!          an_error_code)
!       print *, "koliop_Release_f:an_error_code", an_error_code

!       call koliop_Deinitialize_f(do_kokkos_initialization, an_error_code)
!       print *, "koliop_Deinitialize_f:an_error_code", an_error_code
!    END SUBROUTINE
! END PROGRAM
