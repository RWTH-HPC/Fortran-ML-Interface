module ml_coupling_strategy

    use phydll_f
    use iso_c_binding

    implicit none

    logical, public :: is_phydll_initialized = .false.

    type, abstract :: ml_coupling_strategy_t

        ! "member" variables of the abstract type go here
        character(len=64) :: debug_tag

        contains

        procedure(abstract_ml_coupling_strategy_init), deferred :: ml_coupling_strategy_init
        procedure(abstract_ml_coupling_strategy_inference), deferred :: ml_coupling_strategy_inference
        procedure(abstract_ml_coupling_strategy_finalize), deferred :: ml_coupling_strategy_finalize

        generic :: convert_fields_to_tensor1D => r8_fields_to_r8_tensor1D, r8_fields_to_r4_tensor1D
        procedure :: r8_fields_to_r8_tensor1D
        procedure :: r8_fields_to_r4_tensor1D

        generic :: convert_tensor1D_to_fields => r8_tensor1D_to_r8_fields, r4_tensor1D_to_r8_fields
        procedure :: r8_tensor1D_to_r8_fields
        procedure :: r4_tensor1D_to_r8_fields


        procedure :: set_strategy_debug_tag

    end type

    interface
        subroutine abstract_ml_coupling_strategy_init(self, model, input_shape, output_shape, batch_size, comm)
            import
            class(ml_coupling_strategy_t), intent(inout) :: self

            character(len=256),                 intent(in)    :: model
            integer, dimension(:),              intent(in)    :: input_shape
            integer, dimension(:),              intent(in)    :: output_shape
            integer,                            intent(in)    :: batch_size
            integer,                            intent(inout)    :: comm
        end subroutine

        subroutine abstract_ml_coupling_strategy_inference(self, input_fields, output_fields)
            import
            class(ml_coupling_strategy_t), intent(inout) :: self

            real(8), dimension(:,:,:,:,:), intent(in),  target, contiguous  &
              :: input_fields
            real(8), dimension(:,:,:,:,:), intent(out), target, contiguous  &
              :: output_fields
        end subroutine

        subroutine abstract_ml_coupling_strategy_finalize(self)
            import
            class(ml_coupling_strategy_t), intent(inout) :: self
        end subroutine
    end interface

    contains

    subroutine ml_strategy_mpmd_init(ml_coupling_strategy_id, gcomm, lcomm)
        integer, intent(in)  :: ml_coupling_strategy_id
        integer, intent(in)  :: gcomm ! global communicator (solver + DL)
        integer, intent(out) :: lcomm ! local communicator (only solver)

        character(kind=c_char, len=16) :: instance

        if (ml_coupling_strategy_id == 2 .and. .not. is_phydll_initialized) then
            instance = "physical"
            call phydll_init_f(instance=instance, comm=lcomm)
            
            ! PhyDLL options
            !call phydll_opt_enable_cpl_loop_f()
            !call phydll_opt_set_freq_f(1)
            !call phydll_opt_set_output_freq_f(1)
            is_phydll_initialized = .true.
        endif

    end subroutine

    subroutine r8_fields_to_r8_tensor1D(self, fields, tensor)
        class(ml_coupling_strategy_t), intent(inout) :: self

        real(kind=8), dimension(:,:,:,:,:), intent(in)  :: fields
        real(kind=8), dimension(:),     intent(out) :: tensor

        integer, dimension(1) :: shape_1D
        integer, dimension(5) :: new_shape

        integer :: dim1, dim2, dim3, dim4, dim5
        integer :: b, i, j, k, c
        integer :: index

    
        dim1 = size(fields, 1) ! batch
        dim2 = size(fields, 2) ! x
        dim3 = size(fields, 3) ! y
        dim4 = size(fields, 4) ! z
        dim5 = size(fields, 5) ! channels (scalar/vector)

        do c = 1, dim5
            do k = 1, dim4
                do j = 1, dim3
                    do i = 1, dim2
                        do b = 1, dim1
                            index = (b-1)*dim5*dim2*dim3*dim4 + &
                                    (k-1)*dim5*dim2*dim3 + &
                                    (j-1)*dim5*dim2 + &
                                    (i-1)*dim5 + &
                                    (c-1) &
                                    + 1     ! add +1 because the index calculation is zero based (as in C)
                            tensor(index) = fields(b, i, j, k, c)
                        end do
                    end do
                end do
            end do
        end do

    end subroutine

    subroutine r8_fields_to_r4_tensor1D(self, fields, tensor)
        class(ml_coupling_strategy_t), intent(inout) :: self

        real(kind=8), dimension(:,:,:,:,:), intent(in)  :: fields
        real(kind=4), dimension(:),     intent(out) :: tensor

        integer, dimension(1) :: shape_1D
        integer, dimension(5) :: new_shape

        integer :: dim1, dim2, dim3, dim4, dim5
        integer :: b, i, j, k, c
        integer :: index

    
        dim1 = size(fields, 1) ! rep
        dim2 = size(fields, 2) ! x
        dim3 = size(fields, 3) ! y
        dim4 = size(fields, 4) ! z
        dim5 = size(fields, 5) ! channels (scalar/vector)
        !F fields(channels, x, y, z, rep)
        !C fields[rep][z][y][x][channels]

        do c = 1, dim5
            do k = 1, dim4
                do j = 1, dim3
                    do i = 1, dim2
                        do b = 1, dim1
                            index = (b-1)*dim5*dim2*dim3*dim4 + &
                                    (k-1)*dim5*dim2*dim3 + &
                                    (j-1)*dim5*dim2 + &
                                    (i-1)*dim5 + &
                                    (c-1) &
                                    + 1     ! add +1 because the index calculation is zero based (as in C)
                            tensor(index) = real(fields(b, i, j, k, c), 4)
                        end do
                    end do
                end do
            end do
        end do

    end subroutine

    subroutine r4_tensor1D_to_r8_fields(self, tensor, fields)
        class(ml_coupling_strategy_t), intent(inout) :: self

        real(kind=4), dimension(:),         intent(in)  :: tensor
        real(kind=8), dimension(:,:,:,:,:), intent(out) :: fields

        real(kind=8), dimension(:,:,:,:,:), allocatable :: fields_row_major
        integer, dimension(5) :: fields_shape
        integer :: dim1, dim2, dim3, dim4, dim5
        integer :: b, i, j, k, c
        integer :: index

        dim1 = size(fields, 1)
        dim2 = size(fields, 2)
        dim3 = size(fields, 3)
        dim4 = size(fields, 4)
        dim5 = size(fields, 5)
        write(*,*) "dim1 = ", dim1
        write(*,*) "dim2 = ", dim2
        write(*,*) "dim3 = ", dim3
        write(*,*) "dim4 = ", dim4
        write(*,*) "dim5 = ", dim5

        do c = 1, dim5
            do k = 1, dim4
                do j = 1, dim3
                    do i = 1, dim2
                        do b = 1, dim1
                            index = (b-1)*dim5*dim2*dim3*dim4 + &
                                    (k-1)*dim5*dim2*dim3 + &
                                    (j-1)*dim5*dim2 + &
                                    (i-1)*dim5 + &
                                    (c-1) &
                                    + 1     ! add +1 because the index calculation is zero based (as in C)
                            fields(b, i, j, k, c) = tensor(index)
                        end do
                    end do
                end do
            end do
        end do
    
        fields_shape = shape(fields)

    end subroutine

    subroutine r8_tensor1D_to_r8_fields(self, tensor, fields)
        class(ml_coupling_strategy_t), intent(inout) :: self

        real(kind=8), dimension(:),         intent(in)  :: tensor
        real(kind=8), dimension(:,:,:,:,:), intent(out) :: fields

        real(kind=8), dimension(:,:,:,:,:), allocatable :: fields_row_major
        integer, dimension(5) :: fields_shape
        integer :: dim1, dim2, dim3, dim4, dim5
        integer :: b, i, j, k, c
        integer :: index

        dim1 = size(fields, 1)
        dim2 = size(fields, 2)
        dim3 = size(fields, 3)
        dim4 = size(fields, 4)
        dim5 = size(fields, 5)

        do c = 1, dim5
            do k = 1, dim4
                do j = 1, dim3
                    do i = 1, dim2
                        do b = 1, dim1
                            index = (b-1)*dim5*dim2*dim3*dim4 + &
                                    (k-1)*dim5*dim2*dim3 + &
                                    (j-1)*dim5*dim2 + &
                                    (i-1)*dim5 + &
                                    (c-1) &
                                    + 1     ! add +1 because the index calculation is zero based (as in C)
                            fields(b, i, j, k, c) = tensor(index)
                        end do
                    end do
                end do
            end do
        end do
    end subroutine

    subroutine set_strategy_debug_tag(self, tag)
        class(ml_coupling_strategy_t), intent(inout) :: self
        character(*), intent(in) :: tag

        self%debug_tag = tag
    end subroutine

end module