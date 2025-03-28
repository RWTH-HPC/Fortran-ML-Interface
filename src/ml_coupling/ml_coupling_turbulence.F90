#ifdef SCOREP
#include "scorep/SCOREP_User.inc"
#endif

module ml_coupling_turbulence

    use ml_coupling
    use ml_coupling_strategy
    use ml_coupling_strategy_aix
    use ml_coupling_strategy_phydll

    implicit none

    type, extends(ml_coupling_t) :: ml_coupling_turbulence_t

        integer :: irank
        integer :: upsampling

        ! variables for post processing
        real(kind=8), dimension(:,:,:), allocatable :: field_cpy
        real(kind=8), dimension(:), allocatable :: filter_kernel
        integer :: filter_size

        ! tau_ij computation
        real(kind=8), dimension(:,:,:), allocatable :: uij_sr
        real(kind=8), dimension(:,:,:), allocatable :: uij_filt
        real(kind=8), dimension(:,:,:), allocatable :: ui_filt
        real(kind=8), dimension(:,:,:), allocatable :: uj_filt
        real(kind=8), dimension(:,:,:), allocatable :: tmp_filt
        real(kind=8), dimension(:,:,:), allocatable :: residual_kinetic_energy

        real(kind=8), dimension(:,:,:), allocatable :: u_ds, v_ds, w_ds
        real(kind=8), dimension(:,:,:), allocatable :: prod_ds

        contains

            procedure :: init
            procedure :: parallel_init
            procedure :: preprocess_input
            procedure :: inference
            procedure :: postprocess_output
            procedure :: parallel_finalize
            procedure :: finalize

            ! utility functions
            procedure :: uniform_filtering
            procedure :: downsampling
            procedure :: compute_tau_ij
            procedure :: compute_tau

    end type ml_coupling_turbulence_t

    contains

        subroutine init(self, input_fields, output_fields, model_path, batch_size, input_shape, output_shape, &
            ghost_cells, input_min, input_max, norm_min, norm_max, output_min, &
            output_max, coupling_strategy_id, app_comm)

            class(ml_coupling_turbulence_t), intent(inout) :: self

            real(kind=8), dimension(:,:,:,:), intent(in), target :: input_fields
            real(kind=8), dimension(:,:,:,:), intent(in), target :: output_fields

            character(len=256),             intent(in)  :: model_path
            integer,                        intent(in)  :: batch_size
            integer, dimension(:),          intent(in)  :: input_shape
            integer, dimension(:),          intent(in)  :: output_shape
            integer, dimension(:),          intent(in)  :: ghost_cells

            real(kind=8),                   intent(in)  :: input_min, input_max
            real(kind=8),                   intent(in)  :: norm_min, norm_max
            real(kind=8),                   intent(in)  :: output_min, output_max

            integer,                        intent(in)  :: coupling_strategy_id
            integer,                        intent(in)  :: app_comm

            ! local variables
            integer                                     :: error
            integer :: nx, ny, nz


            self%input_fields => input_fields   
            self%output_fields => output_fields 
            self%model_path = model_path
            self%batch_size = batch_size
            self%input_min = input_min
            self%input_max = input_max
            self%norm_min = norm_min
            self%norm_max = norm_max
            self%output_min = output_min
            self%output_max = output_max

            self%coupling_strategy_id = coupling_strategy_id
            self%app_comm = app_comm

            call MPI_Comm_rank(self%app_comm, self%irank, error)

            ! allocate space for preprocessed input (before inference)
            allocate(self%input_fields_pre( input_shape(1), input_shape(2), &
            input_shape(3), input_shape(4), input_shape(5) ), stat=error)
            if (error /= 0) then
                write(*,*) "ERROR: Could not allocate input_fields_pre!"
            endif

            ! allocate space for output (after inference) to be post-processed
            allocate(self%output_fields_post( output_shape(1), output_shape(2), &
            output_shape(3), output_shape(4), output_shape(5) ), stat=error)
            if (error /= 0) then
                write(*,*) "ERROR: Could not allocate output_fields_post!"
            endif

            ! allocate space for ghost_cells array
            allocate(self%ghost_cells(3), stat=error)
            if (error /= 0) then
                write(*,*) "ERROR: Could not allocate ghost_cells!"
            endif
            self%ghost_cells(1) = ghost_cells(1)
            self%ghost_cells(2) = ghost_cells(2)
            self%ghost_cells(3) = ghost_cells(3)

            ! choose concrete coupling strategy
#ifdef WITH_AIX
            ! AIxeleratorService
            if (self%coupling_strategy_id == 1) then
                allocate(ml_coupling_strategy_aix_t :: self%coupling_strategy)
            endif
#endif
#ifdef WITH_PHYDLL
            ! PhyDLL
            if (self%coupling_strategy_id == 2) then
                allocate(ml_coupling_strategy_phydll_t :: self%coupling_strategy)
            endif
#endif

            call self%coupling_strategy%ml_coupling_strategy_init(self%model_path, input_shape, output_shape, self%batch_size, self%app_comm)

            ! TODO: find a way to generalize this nicely
            !self%upsampling = 2
            self%upsampling = 4
            

            ! setup for postprocessing of tau_ij
            nx = input_shape(2)
            ny = input_shape(3)
            nz = input_shape(4)
            
            self%filter_size = self%upsampling + 1

            allocate(self%field_cpy(nx*self%upsampling, ny*self%upsampling, nz*self%upsampling), source=-13.37_8, stat=error)
            if (error /= 0) then
                write(*,*) "ERROR: could not allocate memory for field copy!"
            endif

            allocate(self%filter_kernel(self%filter_size), source=-13.37_8, stat=error)
            if (error /= 0) then
                write(*,*) "ERROR: could not allocate memory for filter kernel!"
            endif

            allocate(self%uij_sr(nx*self%upsampling, ny*self%upsampling, nz*self%upsampling), source=-13.37_8, stat=error)
            if (error /= 0) then
                write(*,*) "ERROR: could not allocate memory for uij_sr!"
            endif
            
            allocate(self%uij_filt(nx, ny, nz), source=-13.37_8, stat=error)
            if (error /= 0) then
                write(*,*) "ERROR: could not allocate memory for uij_filt!"
            endif

            allocate(self%ui_filt(nx, ny, nz), source=-13.37_8, stat=error)
            if (error /= 0) then
                write(*,*) "ERROR: could not allocate memory for ui_filt!"
            endif

            allocate(self%uj_filt(nx, ny, nz), source=-13.37_8, stat=error)
            if (error /= 0) then
                write(*,*) "ERROR: could not allocate memory for uj_filt!"
            endif

            allocate(self%tmp_filt(nx*self%upsampling, ny*self%upsampling, nz*self%upsampling), source=-13.37_8, stat=error)
            if (error /= 0) then
                write(*,*) "ERROR: could not allocate memory for tmp_filt!"
            endif

            allocate(self%residual_kinetic_energy(nx, ny, nz), source=-13.37_8, stat=error)
            if (error /= 0) then
                write(*,*) "ERROR: could not allocate memory for residual kinetic energy!"
            endif

            allocate(self%u_ds(nx, ny, nz), source=-13.37_8, stat=error)
            if (error /= 0) then
                write(*,*) "ERROR: could not allocate memory for U downsampled!"
            endif

            allocate(self%v_ds(nx, ny, nz), source=-13.37_8, stat=error)
            if (error /= 0) then
                write(*,*) "ERROR: could not allocate memory for V downsampled!"
            endif

            allocate(self%w_ds(nx, ny, nz), source=-13.37_8, stat=error)
            if (error /= 0) then
                write(*,*) "ERROR: could not allocate memory for W downsampled!"
            endif

            allocate(self%prod_ds(nx, ny, nz), source=-13.37_8, stat=error)
            if (error /= 0) then
                write(*,*) "ERROR: could not allocate memory for tmp product downsampled!"
            endif

        end subroutine

        subroutine parallel_init(self, comm)

            class(ml_coupling_turbulence_t), intent(inout) :: self

            integer :: comm  ! MPI communicator of the host simulation code
        end subroutine

        subroutine preprocess_input(self, input_fields, &
            input_fields_pre)

            class(ml_coupling_turbulence_t), intent(inout) :: self

            real(kind=8), dimension(:,:,:,:), intent(in)  :: input_fields
            real(kind=8), dimension(:,:,:,:,:), intent(out), target :: input_fields_pre

            integer :: i, j, k, c
            real(8), pointer :: val_ptr
            real(8) :: clip_max
            real(8), dimension(:,:,:), pointer :: field

            integer :: gx, gy, gz


            gx = self%ghost_cells(1)
            gy = self%ghost_cells(2)
            gz = self%ghost_cells(3)

            do c = 1, size(input_fields, 4)
                input_fields_pre(1,:,:,:,c) = &
                        input_fields(  &
                            1 + gx : size(input_fields_pre, 2) + gx, &
                            1 + gy : size(input_fields_pre, 3) + gy, &
                            1 + gz : size(input_fields_pre, 4) + gz, &
                            c                                        &
                        )    
            end do

            call self%normalize_fields(input_fields_pre)

        end subroutine

        subroutine inference(self, input_fields_pre, output_fields_post)

            class(ml_coupling_turbulence_t), intent(inout) :: self

            real(kind=8), dimension(:,:,:,:,:), intent(in) :: input_fields_pre
            real(kind=8), dimension(:,:,:,:,:), intent(out) :: output_fields_post

            call self%coupling_strategy%ml_coupling_strategy_inference(input_fields_pre, output_fields_post)
        end subroutine

        subroutine postprocess_output(self, output_fields_post, output_fields)

            class(ml_coupling_turbulence_t), intent(inout) :: self

            real(kind=8), dimension(:,:,:,:,:), target, intent(in)  :: output_fields_post
            real(kind=8), dimension(:,:,:,:), target, intent(out) :: output_fields

            real(kind=8), dimension(:,:,:,:), allocatable :: output_fields_denorm
            integer, dimension(3) :: npad
            integer :: c, f
            integer :: i,j,k

            real(kind=8), dimension(:,:,:), pointer :: field
            real(kind=8), dimension(:,:,:), pointer :: u_sr, v_sr, w_sr
            real(kind=8), dimension(:,:,:), pointer :: tau_11, tau_22, tau_33

            integer :: gx, gy, gz
            integer :: nx, ny, nz

            real(kind=8), dimension(:,:,:), allocatable :: tmp, tmp_f
            integer :: err


            gx = self%ghost_cells(1)
            gy = self%ghost_cells(2)
            gz = self%ghost_cells(3)

            if (self%coupling_strategy_id /= 0) then
                nx = size(output_fields_post, 2) / self%upsampling
                ny = size(output_fields_post, 3) / self%upsampling
                nz = size(output_fields_post, 4) / self%upsampling
            else
                nx = size(output_fields_post, 2)
                ny = size(output_fields_post, 3)
                nz = size(output_fields_post, 4)
            endif


            ! 1) denormalization
            call self%denormalize_fields(self%output_fields_post)
            call self%compute_tau(output_fields_post, output_fields)
            

        end subroutine

        subroutine parallel_finalize(self)

            class(ml_coupling_turbulence_t), intent(inout) :: self

        end subroutine

        subroutine finalize(self)

            class(ml_coupling_turbulence_t), intent(inout) :: self
    
            call self%coupling_strategy%ml_coupling_strategy_finalize()
            deallocate(self%coupling_strategy)
            deallocate(self%input_fields_pre)
            deallocate(self%output_fields_post)

            deallocate(self%field_cpy)
            deallocate(self%filter_kernel)

            deallocate(self%uij_sr)
            deallocate(self%uij_filt)
            deallocate(self%ui_filt)
            deallocate(self%uj_filt)
        end subroutine

        subroutine uniform_filtering(self, field, filtered_field)
            class(ml_coupling_turbulence_t), intent(inout) :: self

            real(kind=8), dimension(:,:,:), intent(in) :: field 
            real(kind=8), dimension(:,:,:), intent(out) :: filtered_field
            
            integer :: i, j, k, l, idx, start
            integer :: npad
            integer :: err
            integer :: nx, ny, nz

            ! implement the filter similar to scipy.ndimage.uniform_filter
            ! scipy implements multi-dimensional filter as a sequence of 1D filters

            nx = size(field, 1)
            ny = size(field, 2)
            nz = size(field, 3)

            if (modulo(self%filter_size, 2) .eq. 0) then
                write(*,*) "WARNING: filter size is an even number! We assume odd filter sizes for now!" // "(", self%filter_size, ")" 
            endif 
            npad = self%filter_size / 2

            self%field_cpy(:,:,:) = field(:,:,:)
            
            ! filter x-direction
            do i = 1, nx
                do j = 1, ny
                    do k = 1, nz
                        ! fill kernel values
                        start = j - npad - 1
                        do l = 1, self%filter_size
                            idx = start + l
                            if (idx < 1 .or. idx > ny) then
                                ! n % p returns {0, p-1}, so zero-based indexing
                                ! transform Fortran 1-based indicies to zero based and back
                                idx = modulo(idx - 1, ny) + 1
                            endif
                            self%filter_kernel(l) = self%field_cpy(i,idx,k)
                        end do
                        filtered_field(i,j,k) = sum(self%filter_kernel) / self%filter_size
                    end do
                end do
            end do

            self%field_cpy(:,:,:) = filtered_field(:,:,:)

            ! filter y-direction
            do i = 1, nx
                do j = 1, ny
                    do k = 1, nz
                        ! fill kernel values
                        start = i - npad - 1
                        do l = 1, self%filter_size
                            idx = start + l
                            if (idx < 1 .or. idx > nx) then
                                ! n % p returns {0, p-1}, so zero-based indexing
                                ! transform Fortran 1-based indicies to zero based and back
                                idx = modulo(idx - 1, nx) + 1
                            endif
                            self%filter_kernel(l) = self%field_cpy(idx,j,k)
                        end do
                        filtered_field(i,j,k) = sum(self%filter_kernel) / self%filter_size
                    end do
                end do
            end do

            self%field_cpy(:,:,:) = filtered_field(:,:,:)

            ! filter z-direction
            do i = 1, nx
                do j = 1, ny
                    do k = 1, nz
                        ! fill kernel values
                        start = k - npad - 1
                        do l = 1, self%filter_size
                            idx = start + l
                            ! n % p returns {0, p-1}, so zero-based indexing
                            ! transform Fortran 1-based indicies to zero based and back
                            if (idx < 1 .or. idx > nz) then
                                idx = modulo(idx - 1, nz) + 1
                            endif
                            self%filter_kernel(l) = self%field_cpy(i,j,idx)
                        end do
                        filtered_field(i,j,k) = sum(self%filter_kernel) / self%filter_size
                    end do
                end do
            end do   
            
        end subroutine

        subroutine downsampling(self, field, field_ds, factor)
            class(ml_coupling_turbulence_t), intent(inout) :: self

            real(kind=8), dimension(:,:,:), intent(in) :: field 
            real(kind=8), dimension(:,:,:), intent(out) :: field_ds
            integer, intent(in) :: factor

            field_ds(:,:,:) = field(::factor, ::factor, ::factor)

        end subroutine

        subroutine compute_tau(self, output_fields_post, output_fields)
            class(ml_coupling_turbulence_t), intent(inout) :: self

            real(kind=8), dimension(:,:,:,:,:), intent(in), contiguous, target  :: output_fields_post
            real(kind=8), dimension(:,:,:,:),   intent(out), contiguous, target :: output_fields

            real(kind=8), dimension(:,:,:), pointer :: u_sr, v_sr, w_sr
            real(kind=8), dimension(:,:,:), pointer :: tau_11, tau_12, tau_13
            real(kind=8), dimension(:,:,:), pointer :: tau_22, tau_23
            real(kind=8), dimension(:,:,:), pointer :: tau_33

            integer :: nx, ny, nz
            integer :: gx, gy, gz

#ifdef SCOREP
            SCOREP_USER_REGION_DEFINE( tauij_handle )
            SCOREP_USER_REGION_BEGIN( tauij_handle, "compute_tauij", SCOREP_USER_REGION_TYPE_COMMON)
#endif

            gx = self%ghost_cells(1)
            gy = self%ghost_cells(2)
            gz = self%ghost_cells(3)

            if (self%coupling_strategy_id /= 0) then
                nx = size(output_fields_post, 2) / self%upsampling
                ny = size(output_fields_post, 3) / self%upsampling
                nz = size(output_fields_post, 4) / self%upsampling
            else
                nx = size(output_fields_post, 2)
                ny = size(output_fields_post, 3)
                nz = size(output_fields_post, 4)
            endif

            u_sr => output_fields_post(1,:,:,:,1)
            v_sr => output_fields_post(1,:,:,:,2)
            w_sr => output_fields_post(1,:,:,:,3)

            call self%uniform_filtering(output_fields_post(1,:,:,:,1), self%tmp_filt)
            call self%downsampling(self%tmp_filt, self%u_ds, self%upsampling)

            call self%uniform_filtering(v_sr, self%tmp_filt)
            call self%downsampling(self%tmp_filt, self%v_ds, self%upsampling)

            call self%uniform_filtering(w_sr, self%tmp_filt)
            call self%downsampling(self%tmp_filt, self%w_ds, self%upsampling)


            ! tau_11
            tau_11 => output_fields(    1 + gx : nx + gx, &
                                        1 + gy : ny + gy, &
                                        1 + gz : nz + gz, &
                                        1 &
                                    )
            call self%compute_tau_ij(u_sr, u_sr, self%u_ds, self%u_ds, tau_11)

            ! tau_12
            tau_12 => output_fields(    1 + gx : nx + gx, &
                                        1 + gy : ny + gy, &
                                        1 + gz : nz + gz, &
                                        2 &
                                    )
            call self%compute_tau_ij(u_sr, v_sr, self%u_ds, self%v_ds, tau_12)

            ! tau_13
            tau_13 => output_fields(    1 + gx : nx + gx, &
                                        1 + gy : ny + gy, &
                                        1 + gz : nz + gz, &
                                        3 &
                                    )
            call self%compute_tau_ij(u_sr, w_sr, self%u_ds, self%w_ds, tau_13)

            ! tau_22
            tau_22 => output_fields(    1 + gx : nx + gx, &
                                        1 + gy : ny + gy, &
                                        1 + gz : nz + gz, &
                                        4 &
                                    )
            call self%compute_tau_ij(v_sr, v_sr, self%v_ds, self%v_ds, tau_22)

            ! tau_23
            tau_23 => output_fields(    1 + gx : nx + gx, &
                                        1 + gy : ny + gy, &
                                        1 + gz : nz + gz, &
                                        5 &
                                    )
            call self%compute_tau_ij(v_sr, w_sr, self%v_ds, self%w_ds, tau_23)

            ! tau_33
            tau_33 => output_fields(    1 + gx : nx + gx, &
                                        1 + gy : ny + gy, &
                                        1 + gz : nz + gz, &
                                        6 &
                                    )
            call self%compute_tau_ij(w_sr, w_sr, self%w_ds, self%w_ds, tau_33)

            ! note: for homogeneous isotropic flow sum of the diagonal elements needs to be 1 --> thus kinetic energy is substracted. So this is probably application specific!
            ! TODO: maybe refactor into a flag that the user can set if needed
            self%residual_kinetic_energy(:,:,:) = 0.5 * ((tau_11(:,:,:) + tau_22(:,:,:)) + tau_33(:,:,:))
            write(*,*) "Fortran: sum of residual kinetic energy = ", sum(self%residual_kinetic_energy)
            
            ! note(fabian): it is important to explicitly define 2.0 and 3.0 as real kind=8 to keep accuracy with the reference result from MLLIB !
            tau_11(:,:,:) = tau_11(:,:,:) - (2.0_8 / 3.0_8 * self%residual_kinetic_energy(:,:,:))
            tau_22(:,:,:) = tau_22(:,:,:) - (2.0_8 / 3.0_8 * self%residual_kinetic_energy(:,:,:))
            tau_33(:,:,:) = tau_33(:,:,:) - (2.0_8 / 3.0_8 * self%residual_kinetic_energy(:,:,:))

#ifdef SCOREP
            SCOREP_USER_REGION_END( tauij_handle )
#endif

        end subroutine

        subroutine compute_tau_ij(self, ui_sr, uj_sr, ui_ds, uj_ds, tau_ij)
            class(ml_coupling_turbulence_t), intent(inout) :: self
            
            real(kind=8), dimension(:,:,:), intent(in)  :: ui_sr, uj_sr
            real(kind=8), dimension(:,:,:), intent(in)  :: ui_ds, uj_ds
            real(kind=8), dimension(:,:,:), intent(out) :: tau_ij

            call self%uniform_filtering(ui_sr(:,:,:) * uj_sr(:,:,:), self%tmp_filt)
            call self%downsampling(self%tmp_filt, self%prod_ds, self%upsampling)
            tau_ij(:,:,:) = self%prod_ds(:,:,:) - (ui_ds(:,:,:) * uj_ds(:,:,:))

        end subroutine


end module