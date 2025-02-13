#ifdef SCOREP
#include "scorep/SCOREP_User.inc"
#endif

module ml_coupling_strategy_phydll

    use ml_coupling_strategy
    use iso_c_binding

    implicit none

    type, extends(ml_coupling_strategy_t) :: ml_coupling_strategy_phydll_t

        integer :: input_size, output_size
        character(kind=c_char, len=64) :: phy_label
        character(kind=c_char, len=64) :: dl_label

        !double precision, dimension(:), pointer :: phy_field
        double precision, dimension(:), pointer :: dl_field

        real(8), dimension(:), pointer :: input_data
        real(8), dimension(:), pointer :: output_data

        contains

        procedure :: ml_coupling_strategy_init
        procedure :: ml_coupling_strategy_inference
        procedure :: ml_coupling_strategy_finalize

    end type

    contains

        subroutine ml_coupling_strategy_init(self, model, input_shape, output_shape, batch_size, comm)
            import
            class(ml_coupling_strategy_phydll_t), intent(inout) :: self

            character(len=256),                 intent(in)    :: model
            integer, dimension(:),              intent(in)    :: input_shape
            integer, dimension(:),              intent(in)    :: output_shape
            integer,                            intent(in)    :: batch_size
            integer,                            intent(inout)    :: comm

            character(kind=c_char, len=16) :: instance
            !integer :: lcomm
            integer :: num_fields, field_size
            integer :: error

            allocate(self%input_data( product(input_shape) ), stat=error)
            if (error /= 0) then
                write(*,*) "ERROR: Could not allocate input_data!"
            endif

            allocate(self%output_data( product(output_shape) ), stat=error)
            if (error /= 0) then
                write(*,*) "ERROR: Could not allocate output_data!"
            endif

            ! this has been moved to parallel_init in parallel_m.f90 for now
            if (.not. is_phydll_initialized) then
                instance = "physical"
                call phydll_init_f(instance=instance, comm=comm)

                is_phydll_initialized = .true.
            endif
            
            ! PhyDLL options
            call phydll_opt_enable_cpl_loop_f()
            call phydll_opt_set_freq_f(1)
            call phydll_opt_set_output_freq_f(1)
            

            ! define physical fields
            ! we will define our 5D tensor as one big field
            ! this should be more efficient for MPI communication
            num_fields = 1
            ! additionally we need to send meta information (input/output shapes)
            !num_fields = num_fields + 1
            self%input_size = product(input_shape)
            self%output_size = product(output_shape)
            ! PhyDLL only supports to send and receive fields of same size
            field_size = max(self%input_size, self%output_size)
            call phydll_define_phy_f(num_fields, field_size)

            self%phy_label = "phy_input_field_0"
        end subroutine

        subroutine ml_coupling_strategy_inference(self, input_fields, output_fields)
            import
            class(ml_coupling_strategy_phydll_t), intent(inout) :: self

            real(8), dimension(:,:,:,:,:), intent(in), target, contiguous    :: input_fields
            real(8), dimension(:,:,:,:,:), intent(out), target, contiguous   :: output_fields

            real(8), dimension(:), pointer :: output_ptr
            integer :: i, field_size

#ifdef SCOREP
            SCOREP_USER_REGION_DEFINE( set_field_handle )
            SCOREP_USER_REGION_DEFINE( phydll_send_handle )
            SCOREP_USER_REGION_DEFINE( phydll_recv_handle )
            SCOREP_USER_REGION_DEFINE( get_field_handle )
#endif


            call self%convert_fields_to_tensor1D(input_fields, self%input_data)

#ifdef SCOREP
            SCOREP_USER_REGION_BEGIN( set_field_handle, "ml_coupling_strategy_phydll_set_field", SCOREP_USER_REGION_TYPE_COMMON)
#endif
            !self%phy_field => self%input_data
            call phydll_set_field_f(self%input_data, self%phy_label)
#ifdef SCOREP
            SCOREP_USER_REGION_END( set_field_handle )
            SCOREP_USER_REGION_BEGIN( phydll_send_handle, "ml_coupling_strategy_phydll_send", SCOREP_USER_REGION_TYPE_COMMON)
#endif

            call phydll_isend_f()
            call phydll_wait_isend_f()

#ifdef SCOREP
            SCOREP_USER_REGION_END( phydll_send_handle )
            SCOREP_USER_REGION_BEGIN( phydll_recv_handle, "ml_coupling_strategy_phydll_recv", SCOREP_USER_REGION_TYPE_COMMON)
#endif

            call phydll_irecv_f()
            call phydll_wait_irecv_f()

#ifdef SCOREP
            SCOREP_USER_REGION_END( phydll_recv_handle )
            SCOREP_USER_REGION_BEGIN( get_field_handle, "ml_coupling_strategy_phydll_get_field", SCOREP_USER_REGION_TYPE_COMMON)
#endif

            call phydll_get_field_f(self%dl_field, self%dl_label)
            call phydll_get_field_size_f(field_size)

            !write(*,*) "FORTRAN: dl_field = ", self%dl_field

            if (field_size .le. self%output_size) then
                do i = 1, field_size
                    self%output_data(i) = self%dl_field(i)
                end do
            else
                do i = 1, self%output_size
                    self%output_data(i) = self%dl_field(i)
                end do
            end if
            ! PhyDLL will always reallocate memory to store the dl_field
            ! so we should probably deallocate it after we have copied out the results
            !deallocate(self%dl_field)
#ifdef SCOREP
            SCOREP_USER_REGION_END( get_field_handle )
#endif

            !write(*,*) "output_data = "
            !write(*,*) self%output_data

            call self%convert_tensor1D_to_fields(self%output_data, output_fields)

        end subroutine

        subroutine ml_coupling_strategy_finalize(self)
            import
            class(ml_coupling_strategy_phydll_t), intent(inout) :: self

            deallocate(self%input_data)
            deallocate(self%output_data)
            call phydll_finalize_f()
        end subroutine


end module