// SPDX-License-Identifier: MIT

#ifndef KOLIOP_COLLISION_NIA_H
#define KOLIOP_COLLISION_NIA_H

#include <Kokkos_Macros.hpp>

#include "KOLIOP/internal/configuration.h"
#include "KOLIOP/utility/view.h"

namespace koliop {
    namespace collision {

        ////////////////////////////////////////////////////////////////////////
        // Najl definition
        ////////////////////////////////////////////////////////////////////////

        class Najl {
        public:
            Najl();

            template <typename View6D>
            static void
            Compute(View6D&& the_Najl);

        public:
            // Could be protected but we would have to use friend functions.
            // protected:

            koliop::utility::View::MDD2<ComputeFloatType> C_j_;  // TODO(Etienne M): constant extent.
            koliop::utility::View::MDD3<ComputeFloatType> C_jl_; // TODO(Etienne M): constant extent.
        };


        ////////////////////////////////////////////////////////////////////////
        // Najl methods definition
        ////////////////////////////////////////////////////////////////////////


        // !===========================================================================
        //   !> Computation of I(n,i, Xtr) = int from 0 to infinity of
        //   !>   dU U^i exp(-U) / (U+Xtr)^(n+0.5)
        //   !>  using the recursive computation of
        //   !>   J(k, X_tr) = \int_sqrt(X_tr)^\infty 2 v^(2k) exp(-v**2) dv
        //   !> where k is between -1 and index_max+1
        //   !> WARNING: in the code, the index k in J_vec is shifted to start from 0
        //   !---------------------------------------------------------------------------
        // !DIR$ ATTRIBUTES INLINE :: libcoll_I
        // subroutine libcoll_I( Nspace, sqrtXtr, sqrtXtr_inv, Xtr, offset_k, I_mat )
        //     integer              , intent(in)  :: Nspace
        //     real(F64), contiguous, intent(in)  :: sqrtXtr    (0:,0:)
        //     real(F64), contiguous, intent(in)  :: sqrtXtr_inv(0:,0:)
        //     real(F64), contiguous, intent(in)  :: Xtr        (0:,0:)
        //     integer              , intent(in)  :: offset_k
        //     real(F64), contiguous, intent(out) :: I_mat      (0:,0:,0:)

        //     real(F64) :: J_vec     (0:Nvpar_,0:Nspace,0:index_max_+2)
        //     real(F64) :: Xtr_pows  (0:Nvpar_,0:Nspace,0:index_max_+1)

        //     integer   :: ind_i, ind_k, ivpar

        //     Xtr_pows(:,:,0) = 1._F64

        //     do ind_i = 1, index_max_+1
        //         Xtr_pows(:,:,ind_i) = - Xtr(:,:) * Xtr_pows(:,:,ind_i-1)
        //     end do

        //     J_vec(:,:,1) = sqrtPI * erfc(sqrtXtr(:,:))
        //     J_vec(:,:,0) = 2._F64 * (sqrtXtr_inv(:,:) / exp(Xtr(:,:)) -
        //                             sqrtPI * erfc(sqrtXtr(:,:)))

        //     do ind_k = 2, index_max_+2
        //         J_vec(:,:,ind_k) = (sqrtXtr_inv(:,:) / exp(Xtr(:,:))) * abs(Xtr_pows(:,:,ind_k-1))
        //     end do

        //     do ind_k = 2, index_max_+2
        //         J_vec(:,:,ind_k) += J_vec(:,:,ind_k-1) *
        //                             (ind_k - 1.5_F64)
        //     end do

        //     I_mat(:,:,:) = 0._F64

        //     do ind_i = 0, index_max_+1
        //         do ind_k = 0, ind_i
        //             I_mat(:,:,ind_i) += comb_mat_(ind_k,ind_i) *
        //                                 J_vec(:,:,ind_k-offset_k+1) *
        //                                 Xtr_pows(:,:,ind_i-ind_k) *
        //                                 exp(Xtr(:,:))
        //         end do
        //     end do
        //     end subroutine libcoll_I

        // !===========================================================================
        // !> Computation of L(n,i, sp_a, sp_b) with
        // !>   L(0,i, sp_a, sp_b) = <D_dab / vTa**2 | u^i>
        // !>   L(1,i, sp_a, sp_b) = <1/v d(D_dab)/dv | u^i>
        // !>   L(2,i, sp_a, sp_b) = <nu_vab - nu_dab | u^i>
        // !>   L(3,i, sp_a, sp_b) = <0.5 * v * d(nu_vab - nu_dab)/dv | u^i>
        // !>   L(4,i, sp_a, sp_b) = <1 / v * d(nu_vab - nu_dab)/dv | u^i>
        // !> Note : speciesa is implicitely pSpecies_id_
        // !> Output: L_mat(0:4, 0:index_max_+1, 0:Nvpar_, 0:Nspecies_-1)
        // !---------------------------------------------------------------------------
        // subroutine libcoll_L( Nspace, n_allspecies, T_allspecies, &
        //     nu, vpar2_on_Ta, L_mat )

        //   integer              , intent(in)  :: Nspace
        //   real(F64), contiguous, intent(in)  :: n_allspecies(0:,0:)
        //   real(F64), contiguous, intent(in)  :: T_allspecies(0:,0:)
        //   real(F64), contiguous, intent(in)  :: nu(0:)
        //   real(F64), contiguous, intent(in)  :: vpar2_on_Ta(0:,0:)
        //   real(F64), contiguous, intent(out) :: L_mat(0:,0:,0:,0:)

        //   real(F64) :: I_mat      (0:Nvpar_,0:Nspace,0:index_max_+1)
        //   real(F64) :: Xtr        (0:Nvpar_,0:Nspace)
        //   real(F64) :: sqrtXtr    (0:Nvpar_,0:Nspace)
        //   real(F64) :: sqrtXtr_inv(0:Nvpar_,0:Nspace)
        //   real(F64) :: vTb_on_vta_2(0:Nspace)
        //   real(F64) :: coeff_L(0:Nspace)

        //   integer   :: ispace, ind_i, speciesb
        //   real(F64) :: hat_Aa, hat_Ab
        //   real(F64) :: hat_Za, hat_Zb

        //   hat_Aa  = hat_As_(pSpecies_id_)
        //   hat_Za = hat_Zs_(pSpecies_id_)

        //   L_mat(:,:,:,:) = 0._F64

        //   do speciesb = 0, Nspecies_-1
        //     if ( collision_interspecies_ .or. (pSpecies_id_.eq.speciesb) ) then
        //       hat_Ab = hat_As_(speciesb)
        //       hat_Zb = hat_Zs_(speciesb)

        //       coeff_L(:) = 0.75_F64*sqrtPI*sqrt(2._F64)*nu(:) * ( hat_Zb * hat_Zb * n_allspecies(:,speciesb) ) / ( hat_Za * hat_Za * n_allspecies(:,pSpecies_id_) )

        //       vTb_on_vta_2(:) = hat_Aa * T_allspecies(:,speciesb) / ( hat_Ab * T_allspecies(:,pSpecies_id_) )

        //       !->  Computation of sqrtXtr and I_mat (needed for L_mat(...,0,...))
        //       do ispace = 0,Nspace
        //         Xtr(:,ispace) = vpar2_on_Ta(:,ispace)/2._F64 +
        //                         vTb_on_vta_2(ispace) * (9._F64 * PI / 16._F64)
        //       end do

        //       sqrtXtr(:,:)     = sqrt(Xtr(:,:))
        //       sqrtXtr_inv(:,:) = 1._F64/sqrtXtr(:,:)

        //       call libcoll_I( Nspace, sqrtXtr, sqrtXtr_inv, Xtr, 0, I_mat )

        //       !-> Computation of L_mat(...,0,...)
        //       do ind_i = 0,index_max_
        //         do ispace = 0,Nspace ! explicit because of coeff_L
        //           L_mat(:,ispace,ind_i,0) = L_mat(:,ispace,ind_i,0) +
        //                                     coeff_L(ispace) * I_mat(:,ispace,ind_i)
        //         end do
        //       end do

        //       !-> Computation of L_mat(...,1,...)
        //       do ispace = 0,Nspace ! explicit because of coeff_L
        //         L_mat(:,ispace,0,1) = L_mat(:,ispace,0,1) -
        //                               coeff_L(ispace) * sqrtXtr_inv(:,ispace)
        //       end do

        //       !->  Computation of sqrtXtr and I_mat (needed for L_mat(...,2,...))
        //       do ispace = 0,Nspace
        //         Xtr(:,ispace) = vpar2_on_Ta(:,ispace)/2._F64 +
        //                         2.1_F64 * vTb_on_vta_2(ispace)
        //       end do

        //       sqrtXtr(:,:)     = sqrt(Xtr(:,:))
        //       sqrtXtr_inv(:,:) = 1._F64/sqrtXtr(:,:)

        //       call libcoll_I( Nspace, sqrtXtr, sqrtXtr_inv, Xtr, 1, I_mat )

        //       !-> Computation of L_mat(...,2,...)
        //       do ind_i = 0,index_max_
        //         do ispace = 0,Nspace ! explicit because of coeff_L
        //           L_mat(:,ispace,ind_i,2) += - coeff_L(ispace) * I_mat(:,ispace,ind_i)
        //         end do
        //       end do

        //       !-> Computation of L_mat(...,3,...)
        //       do ispace = 0,Nspace ! explicit because of coeff_L
        //         L_mat(:,ispace,0,3) += coeff_L(ispace) * sqrtXtr_inv(:,ispace)**3 * vpar2_on_Ta(:,ispace) / 2._F64
        //       end do

        //       !-> Computation of L_mat(...,4,...)
        //       do ispace = 0,Nspace ! explicit because of coeff_L
        //         L_mat(:,ispace,0,4) += coeff_L(ispace) * sqrtXtr_inv(:,ispace)**3
        //       end do

        //     end if ! collision_interspecies_ .or. (pSpecies_id_.eq.speciesb)
        //   end do ! speciesb = 0, Nspecies_-1

        //   !-> Computation of L_mat(...,1,...)
        //   L_mat(:,:,0,1) = L_mat(:,:,0,1) + L_mat(:,:,0,0)
        //   do ind_i = 1,index_max_
        //     L_mat(:,:,ind_i,1) += L_mat(:,:,ind_i,0) - L_mat(:,:,ind_i-1,0) * ind_i
        //   end do

        //   !-> Computation of L_mat(...,3,...)
        //   L_mat(:,:,0,3) += L_mat(:,:,0,2) * vpar2_on_Ta(:,:) / 2._F64 + L_mat(:,:,1,2) - L_mat(:,:,0,2)

        //   do ind_i = 1,index_max_
        //     L_mat(:,:,ind_i,3) += - L_mat(:,:,ind_i-1,2) * ind_i * vpar2_on_Ta/2._F64 +
        //                             L_mat(:,:,ind_i  ,2) * (vpar2_on_Ta/2._F64-ind_i-1) +
        //                             L_mat(:,:,ind_i+1,2)
        //   end do

        //   !-> Computation of L_mat(...,4,...)
        //   L_mat(:,:,0,4) = L_mat(:,:,0,4) + L_mat(:,:,0,2)
        //   do ind_i = 1,index_max_
        //     L_mat(:,:,ind_i,4) = L_mat(:,:,ind_i,4) + &
        //       L_mat(:,:,ind_i  ,2) - &
        //       L_mat(:,:,ind_i-1,2) * ind_i
        //   end do

        // end subroutine libcoll_L
        // !---------------------------------------------------------------------------


        // !===========================================================================
        // !> Computation of the N_i,a ^ jl = sum_b M_i,ab ^ jl
        // !---------------------------------------------------------------------------
        // subroutine libcoll_N_a_jl( Nspace, Npol, vparg, C_jl, &
        //     n_allspecies, T_allspecies, nu, N0, N1, N2 )

        //   integer              , intent(in)  :: Nspace
        //   integer              , intent(in)  :: Npol
        //   real(F64), contiguous, intent(in)  :: vparg(0:)
        //   real(F64), contiguous, intent(in)  :: C_jl(0:,0:,0:)
        //   real(F64), contiguous, intent(in)  :: n_allspecies(0:,0:)
        //   real(F64), contiguous, intent(in)  :: T_allspecies(0:,0:)
        //   real(F64), contiguous, intent(in)  :: nu(0:)
        //   real(F64), contiguous, intent(out) :: N0(0:,0:,0:,0:)
        //   real(F64), contiguous, intent(out) :: N1(0:,0:,0:,0:)
        //   real(F64), contiguous, intent(out) :: N2(0:,0:,0:,0:)

        //   real(F64) :: L_mat      (0:Nvpar_,0:Nspace,0:index_max_+1,0:4)
        //   real(F64) :: N_coeff    (0:Nvpar_,0:Nspace)
        //   real(F64) :: vpar2_on_Ta(0:Nvpar_,0:Nspace)

        //   integer :: ispace, ispecies
        //   integer :: j, l, ivpar, ind_i

        //   N0(:,:,:,:) = 0._F64
        //   N1(:,:,:,:) = 0._F64
        //   N2(:,:,:,:) = 0._F64

        //   do ispace = 0,Nspace ! explicit because of vparg
        //     vpar2_on_Ta(:,ispace) = vparg(:)** 2/T_allspecies(ispace,pSpecies_id_)
        //   end do

        //   ! Computation of L(n, i, sp_a, sp_b)
        //   call libcoll_L( Nspace, n_allspecies, T_allspecies, nu, &
        //       vpar2_on_Ta, L_mat )

        //   do ind_i = 0,index_max_

        //     ! Computing N0

        //     N_coeff(:,:) = L_mat(:,:,ind_i,1) * 2._F64 +
        //                    L_mat(:,:,ind_i,2) * (3._F64 - vpar2_on_Ta(:,:)) +
        //                    L_mat(:,:,ind_i,4) * (2._F64 + vpar2_on_Ta(:,:))

        //     if( ind_i.ne.index_max_ ) then
        //       do l = 0,Npol-1
        //         do j = 1,Npol-1
        //           N0(:,:,j,l) += - j * N_coeff(:,:) * C_jl(ind_i,j-1,l)
        //         end do
        //       end do
        //     end if

        //     N_coeff(:,:) = N_coeff(:,:) -
        //                    L_mat(:,:,ind_i  ,0) * 2._F64 -
        //                    L_mat(:,:,ind_i+1,2) * 2._F64

        //     do l = 0,Npol-1
        //       do j = 1,Npol-1
        //         N0(:,:,j,l) = N0(:,:,j,l) + j * N_coeff(:,:) * C_jl(ind_i,j,l)
        //       end do
        //     end do

        //     ! Computing N1

        //     N_coeff(:,:) = - L_mat(:,:,ind_i  ,0) +
        //                      L_mat(:,:,ind_i  ,1) -
        //                      L_mat(:,:,ind_i+1,2) +
        //                      L_mat(:,:,ind_i  ,2) * (2._F64-vpar2_on_Ta(:,:)/2._F64) +
        //                      L_mat(:,:,ind_i  ,3)

        //     if ( ind_i.ne.index_max_ ) then

        //       do l = 0,Npol-1
        //         do ispace = 0,Nspace ! explicit because of vparg
        //           N1(:,ispace,0,l) = N1(:,ispace,0,l) + &
        //               vparg(:) * C_jl(ind_i,0,l) * N_coeff(:,ispace)
        //         end do

        //         do j = 1,Npol-1
        //           do ispace = 0,Nspace ! explicit because of vparg
        //             N1(:,ispace,j,l) += vparg(:) * (C_jl(ind_i,j,l) * N_coeff(:,ispace) +
        //                                             2 * j * L_mat(:,ispace,ind_i,2) * (C_jl(ind_i,j,l)-C_jl(ind_i,j-1,l)))
        //           end do
        //         end do
        //       end do

        //     else

        //       do l = 0,Npol-1
        //         do j = 0,Npol-1
        //           do ispace = 0,Nspace ! explicit because of vparg
        //             N1(:,ispace,j,l) += vparg(:) * C_jl(index_max_,j,l) * (N_coeff(:,ispace) + 2 * j * L_mat(:,ispace,index_max_,2) )
        //           end do
        //         end do
        //       end do

        //     end if ! ind_i.ne.index_max_

        //     ! Computing N2

        //     do ispace = 0,Nspace ! explicit because of T_allspecies
        //       N_coeff(:,ispace) = T_allspecies(ispace,pSpecies_id_) * (L_mat(:,ispace,ind_i,0) + L_mat(:,ispace,ind_i,2) * vpar2_on_Ta(:,ispace)/2._F64 )
        //     end do

        //     do l = 0,Npol-1
        //       do j = 0,Npol-1
        //         N2(:,:,j,l) += C_jl(ind_i,j,l) * N_coeff(:,:)
        //       end do
        //     end do

        //   end do ! ind_i = 0,index_max_

        // end subroutine libcoll_N_a_jl


    } // namespace collision
} // namespace koliop

#endif
