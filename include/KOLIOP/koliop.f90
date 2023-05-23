! SPDX-License-Identifier: MIT

MODULE koliop

   USE ISO_C_Binding

   IMPLICIT NONE

   PRIVATE

   INTEGER, PARAMETER :: koliop_Status = C_INT32_T

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
         IMPORT :: C_DOUBLE
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
   END INTERFACE

   PUBLIC :: koliop_Status
   PUBLIC :: koliop_Initialize
   PUBLIC :: koliop_Deinitialize

CONTAINS

   SUBROUTINE koliop_Initialize( &
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

   SUBROUTINE koliop_Deinitialize( &
      an_error_code)

      IMPLICIT NONE

      INTEGER(koliop_Status), INTENT(OUT) :: an_error_code

      an_error_code = koliop_Deinitialize_c()
   END SUBROUTINE
END MODULE

program test
   USE koliop
   USE ISO_C_Binding

   implicit none
   INTEGER :: a_host_thread_count
   INTEGER :: a_device_index
   LOGICAL :: with_verbose_initialization
   INTEGER(koliop_Status) :: an_error_code
   INTEGER(C_SIZE_T) :: test1

   a_host_thread_count = 8
   a_device_index = 0
   with_verbose_initialization = .TRUE.

   call koliop_Initialize(a_host_thread_count, a_device_index, with_verbose_initialization, an_error_code)


   call koliop_Deinitialize(an_error_code)

   test1 = INT(a_device_index, C_SIZE_T)

   print *, "hi", an_error_code
end program
