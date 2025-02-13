module ml_coupling_combustion
    
    use ml_coupling
    use ml_coupling_strategy
    use ml_coupling_strategy_aix
    use ml_coupling_strategy_phydll

    implicit none

    type, extends(ml_coupling_t) :: ml_coupling_combustion_t

        integer :: irank
        
        contains

            procedure :: init
            procedure :: parallel_init
            procedure :: preprocess_input
            procedure :: inference
            procedure :: postprocess_output
            procedure :: parallel_finalize
            procedure :: finalize

    end type ml_coupling_combustion_t

    contains

        subroutine init(self, input_fields, output_fields, model_path, batch_size, input_shape, output_shape, &
             ghost_cells, input_min, input_max, norm_min, norm_max, output_min, &
              output_max, coupling_strategy_id, app_comm)

            class(ml_coupling_combustion_t), intent(inout) :: self

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

        end subroutine

        subroutine parallel_init(self, comm)

            class(ml_coupling_combustion_t), intent(inout) :: self
    
            integer :: comm  ! MPI communicator of the host simulation code
        end subroutine

        subroutine preprocess_input(self, input_fields, &
             input_fields_pre)

            class(ml_coupling_combustion_t), intent(inout) :: self

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

            class(ml_coupling_combustion_t), intent(inout) :: self

            real(kind=8), dimension(:,:,:,:,:), intent(in) :: input_fields_pre
            real(kind=8), dimension(:,:,:,:,:), intent(out) :: output_fields_post

            
            call self%coupling_strategy%ml_coupling_strategy_inference(input_fields_pre, output_fields_post)
        end subroutine

        subroutine postprocess_output(self, output_fields_post, output_fields)

            class(ml_coupling_combustion_t), intent(inout) :: self

            real(kind=8), dimension(:,:,:,:,:), target, intent(in)  :: output_fields_post
            !type(field_ptr), dimension(:), intent(out) :: output_fields
            real(kind=8), dimension(:,:,:,:), target, intent(out) :: output_fields

            real(kind=8), dimension(:,:,:,:), allocatable :: output_fields_denorm
            integer, dimension(3) :: npad
            integer :: c, f
            integer :: i,j,k

            real(kind=8), dimension(:,:,:), pointer :: field

            integer :: gx, gy, gz


            gx = self%ghost_cells(1)
            gy = self%ghost_cells(2)
            gz = self%ghost_cells(3)

            ! 1) denormalization
            call self%denormalize_fields(self%output_fields_post)
            

            ! TODO(Fabian): add clipping here self%output_min <= output_fields_post(:) <= self%output_max

            do c = 1, size(output_fields, 4)
                output_fields(:,:,:,c) = 0.0_8
                output_fields( &
                    1 + gx : size(output_fields_post, 2) + gx, &
                    1 + gy : size(output_fields_post, 3) + gy, &
                    1 + gz : size(output_fields_post, 4) + gz, &
                    c &
                ) = output_fields_post(1,:,:,:,c)    
            end do

        end subroutine

        subroutine parallel_finalize(self)

            class(ml_coupling_combustion_t), intent(inout) :: self
    
        end subroutine
    
        subroutine finalize(self)

            class(ml_coupling_combustion_t), intent(inout) :: self
    
            call self%coupling_strategy%ml_coupling_strategy_finalize()
            deallocate(self%coupling_strategy)
            deallocate(self%input_fields_pre)
            deallocate(self%output_fields_post)
        end subroutine



end module