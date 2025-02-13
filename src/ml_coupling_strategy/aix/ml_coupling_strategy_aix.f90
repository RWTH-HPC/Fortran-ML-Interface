module ml_coupling_strategy_aix

    use ml_coupling_strategy
    use aixelerator_service_mod
    use iso_c_binding, only: C_ptr, c_null_char, C_int64_t

    implicit none

    type, extends(ml_coupling_strategy_t) :: ml_coupling_strategy_aix_t

        ! "member" variables of the type go here
        type(C_ptr) :: aixelerator ! TODO(fabian): do we need the safe keyword here?

        real(4), dimension(:), allocatable :: input_data, output_data

        contains

        procedure :: ml_coupling_strategy_init
        procedure :: ml_coupling_strategy_inference
        procedure :: ml_coupling_strategy_finalize

    end type

    contains

        subroutine ml_coupling_strategy_init(self, model, input_shape, output_shape, batch_size, comm)
            import
            class(ml_coupling_strategy_aix_t), intent(inout) :: self

            character(len=256),                 intent(in)    :: model
            integer, dimension(:),              intent(in)    :: input_shape
            integer, dimension(:),              intent(in)    :: output_shape
            integer,                            intent(in)    :: batch_size
            integer,                            intent(inout) :: comm

            integer                                           :: error
            character(len=256)                                :: model_c
            integer(C_int64_t), dimension(:), allocatable     :: input_shape_c, output_shape_c
            integer :: input_shape_c_size, output_shape_c_size

            allocate(self%input_data( product(input_shape) ), stat=error)
            if (error /= 0) then
                write(*,*) "ERROR: Could not allocate input_data!"
            endif
            !write(*,*) "Allocated input data with size = ", product(input_shape)

            allocate(self%output_data( product(output_shape) ), stat=error)
            if (error /= 0) then
                write(*,*) "ERROR: Could not allocate output_data!"
            endif
            !write(*,*) "Allocated output data with size = ", product(output_shape)

            allocate(input_shape_c(input_shape_c_size), stat=error)
            if (error /= 0) then
                write(*,*) "ERROR: Could not allocate input_shape_C! Error code = ", error
            endif
            ! check if input fields are actually only 2D
            if (input_shape(4) == 1) then
                input_shape_c_size = 4
                input_shape_c = [input_shape(1), input_shape(2), input_shape(3), input_shape(5)]
            else
                input_shape_c_size = 5
                input_shape_c = [input_shape(1), input_shape(2), input_shape(3), input_shape(4), input_shape(5)]
            endif
            
            allocate(output_shape_c(output_shape_c_size), stat=error)
            if (error /= 0) then
                write(*,*) "ERROR: Could not allocate output_shape_C"
            endif
            ! check if output fields are actually only 2D
            if (output_shape(4) == 1) then
                output_shape_c_size = 4
                output_shape_c = [output_shape(1), output_shape(2), output_shape(3), output_shape(5)]
            else
                output_shape_c_size = 5
                output_shape_c = [output_shape(1), output_shape(2), output_shape(3), output_shape(4), output_shape(5)]
            endif

            model_c = trim(model) // c_null_char
            !input_shape_c = [ input_shape(1), input_shape(2), input_shape(3), input_shape(4), input_shape(5) ]
            !output_shape_c = [ output_shape(1), output_shape(2), output_shape(3), output_shape(4), output_shape(5) ]
            self%aixelerator = createAIxeleratorServiceFloat_C(model_c, input_shape_c, size(input_shape_c), self%input_data, output_shape_c, size(output_shape_c), self%output_data, batch_size, comm)

        end subroutine

        subroutine ml_coupling_strategy_inference(self, input_fields, output_fields)
            import
            class(ml_coupling_strategy_aix_t), intent(inout) :: self

            real(8), dimension(:,:,:,:,:), intent(in),  target    :: input_fields
            real(8), dimension(:,:,:,:,:), intent(out), target    :: output_fields

            integer :: i

            call self%convert_fields_to_tensor1D(input_fields, self%input_data)

            call inferenceAIxeleratorServiceFloat_C(self%aixelerator)

            call self%convert_tensor1D_to_fields(self%output_data, output_fields)
        end subroutine

        subroutine ml_coupling_strategy_finalize(self)
            import
            class(ml_coupling_strategy_aix_t), intent(inout) :: self

            call deleteAIxeleratorServiceFloat_C(self%aixelerator)      
            deallocate(self%input_data)
            deallocate(self%output_data)
        end subroutine

end module