! SPDX-License-Identifier: MIT

MODULE koliop

   USE ISO_C_Binding

   IMPLICIT NONE

   PRIVATE

   PUBLIC :: koliop_Status
   PUBLIC :: koliop_Initialize

   INTEGER, PARAMETER :: koliop_Status = C_INT32_T

   INTERFACE
      ! koliop_Status koliop_Initialize(size_t a_host_thread_count,
      !                              size_t a_device_index,
      !                              int8_t with_verbose_initialization);
      INTEGER(koliop_Status) FUNCTION koliop_Initialize_c( &
         a_host_thread_count, &
         a_device_index, &
         with_verbose_initialization) &
         BIND(C, NAME='koliop_Initialize')

         IMPORT :: C_INT8_T
         IMPORT :: C_SIZE_T
         IMPORT :: koliop_Status

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
   END INTERFACE

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

   test1 = INT(a_device_index, C_SIZE_T)

   print *, "hi", an_error_code
end program
