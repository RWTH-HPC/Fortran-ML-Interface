#ifdef SCOREP
#include "scorep/SCOREP_User.inc"
#endif

module ml_coupling

    use ml_coupling_strategy

    implicit none

    ! Derived type to construct an array of pointers with
    ! Will be used to store multiple 3D input fields as one 4D field
    type :: field_ptr
        real(kind=8), dimension(:,:,:), pointer :: field
    end type field_ptr

    ! definition of the abstract interface
    ! defines which abstract routines are deferred
    type, abstract :: ml_coupling_t

        ! "member" variables of the base type
        !type(field_ptr), dimension(:), pointer :: input_fields
        !type(field_ptr), dimension(:), pointer :: output_fields 
        real(kind=8), dimension(:,:,:,:), pointer :: input_fields
        real(kind=8), dimension(:,:,:,:), pointer :: output_fields 

        real(kind=8), dimension(:,:,:,:,:), allocatable :: input_fields_pre
        real(kind=8), dimension(:,:,:,:,:), allocatable :: output_fields_post
        integer,      dimension(:),         allocatable :: ghost_cells

        ! variables for normalization
        real(kind=8)                    :: input_min, input_max
        real(kind=8)                    :: norm_min, norm_max
        real(kind=8)                    :: output_min, output_max

        class(ml_coupling_strategy_t), pointer :: coupling_strategy
        integer :: coupling_strategy_id

        character(256) :: model_path
        integer :: batch_size
        integer :: app_comm

        contains

        procedure(abstract_init), deferred :: init
        ! hook to be called from CIAO's parallel_init in src/library/parallel_m.f90
        procedure(abstract_parallel_init), deferred :: parallel_init

        procedure(abstract_preprocess_input), deferred :: preprocess_input
        procedure(abstract_inference), deferred :: inference
        procedure(abstract_postprocess_output), deferred :: postprocess_output

        ! hook to be called from CIAO's parallel_final in src/library/parallel_m.f90
        procedure(abstract_parallel_finalize), deferred :: parallel_finalize
        procedure(abstract_finalize), deferred :: finalize

        ! utility routines
        procedure :: normalize_2D
        procedure :: normalize_3D
        generic :: normalize => normalize_2D, normalize_3D

        procedure :: normalize_fields

        procedure :: denormalize_2D
        procedure :: denormalize_3D
        generic :: denormalize => denormalize_2D, denormalize_3D

        procedure :: denormalize_fields

        procedure:: ml_step

    end type ml_coupling_t

    ! interface definition for the deferred routines
    ! define the parameters passed to the subroutines
    abstract interface

        subroutine abstract_init(self, input_fields, output_fields, model_path, batch_size, input_shape, output_shape, &
            ghost_cells, input_min, input_max, norm_min, norm_max, output_min, output_max, coupling_strategy_id, app_comm)
            import
            class(ml_coupling_t), intent(inout) :: self

            !type(field_ptr), dimension(:), intent(in), target :: input_fields
            !type(field_ptr), dimension(:), intent(in), target :: output_fields
            real(kind=8), dimension(:,:,:,:), intent(in), target :: input_fields
            real(kind=8), dimension(:,:,:,:), intent(in), target :: output_fields

            character(256),                   intent(in)  :: model_path
            integer,                          intent(in)  :: batch_size
            integer,      dimension(:),       intent(in)  :: input_shape
            integer,      dimension(:),       intent(in)  :: output_shape
            integer,      dimension(:),       intent(in)  :: ghost_cells

            real(kind=8),                     intent(in)  :: input_min, input_max
            real(kind=8),                     intent(in)  :: norm_min, norm_max
            real(kind=8),                     intent(in)  :: output_min, output_max

            integer,                          intent(in)  :: coupling_strategy_id
            integer,                          intent(in)  :: app_comm
        end subroutine

        subroutine abstract_parallel_init(self, comm)
            import 
            class(ml_coupling_t), intent(inout) :: self

            integer :: comm  ! MPI communicator of the host simulation code
        end subroutine

        subroutine abstract_preprocess_input(self, input_fields, &
            input_fields_pre)
            import
            class(ml_coupling_t), intent(inout) :: self

            !type(field_ptr), dimension(:), intent(in)  :: input_fields
            real(kind=8), dimension(:,:,:,:), intent(in)  :: input_fields
            real(kind=8), dimension(:,:,:,:,:), intent(out), target :: input_fields_pre
        end subroutine

        subroutine abstract_inference(self, input_fields_pre, output_fields_post)
            import
            class(ml_coupling_t), intent(inout) :: self

            real(kind=8), dimension(:,:,:,:,:), intent(in)  :: input_fields_pre
            real(kind=8), dimension(:,:,:,:,:), intent(out) :: output_fields_post
        end subroutine

        subroutine abstract_postprocess_output(self, output_fields_post, output_fields)
            import
            class(ml_coupling_t), intent(inout) :: self

            real(kind=8), dimension(:,:,:,:,:), target, intent(in)  :: output_fields_post
            !type(field_ptr), dimension(:), intent(out) :: output_fields
            real(kind=8), dimension(:,:,:,:), target, intent(out) :: output_fields
        end subroutine

        subroutine abstract_parallel_finalize(self)
            import 
            class(ml_coupling_t), intent(inout) :: self

        end subroutine

        subroutine abstract_finalize(self)
            import 
            class(ml_coupling_t), intent(inout) :: self

        end subroutine

    end interface

    ! subroutines of the base module that might be usable for all concrete implementations without the need for overwriting them
    contains

        subroutine ml_coupling_mpmd_init(ml_coupling_strategy_id, gcomm, lcomm)
            integer, intent(in)  :: ml_coupling_strategy_id
            integer, intent(in)  :: gcomm ! global communicator (solver + DL)
            integer, intent(out) :: lcomm ! local communicator (only solver)

            call ml_strategy_mpmd_init(ml_coupling_strategy_id, gcomm, lcomm)
        end subroutine

        ! TODO(fabian): maybe we can refactor both normalization routines into a single one 
        ! that gets passed a pointer and interprets the pointer as a 1D array to iterate over it
        subroutine normalize_2D(self, x, xmin, xmax, a, b, xnorm)
            class(ml_coupling_t) :: self

            real(kind=8), dimension(:,:), intent(in)  :: x
            real(kind=8),                 intent(in)  :: xmin, xmax, a, b
            real(kind=8), dimension(:,:), intent(out) :: xnorm

            xnorm = a + (b-a) * (x(:,:) - xmin) / (xmax - xmin)
        end subroutine

        subroutine normalize_3D(self, x, xmin, xmax, a, b, xnorm)
            class(ml_coupling_t) :: self 
            
            real(kind=8), dimension(:,:,:), intent(in)  :: x
            real(kind=8),                   intent(in)  :: xmin, xmax, a, b
            real(kind=8), dimension(:,:,:), intent(out) :: xnorm

            xnorm = a + (b-a) * (x(:,:,:) - xmin) / (xmax - xmin)
        end subroutine

        subroutine normalize_fields(self, input_fields_pre)
            class(ml_coupling_t) :: self

            real(kind=8), dimension(:,:,:,:,:), intent(inout), target :: input_fields_pre

            integer :: b, c
            real(kind=8), dimension(:,:,:), pointer :: sub_field_3d

#ifdef SCOREP
            SCOREP_USER_REGION_DEFINE( normalize_fields_handle )
            SCOREP_USER_REGION_BEGIN( normalize_fields_handle, "normalize_fields", SCOREP_USER_REGION_TYPE_COMMON)
#endif

            do b = 1, size(input_fields_pre, 1)
                do c = 1, size(input_fields_pre, 5)
                    sub_field_3d => input_fields_pre(b,:,:,:,c)
                    call self%normalize(sub_field_3d, self%input_min, &
                    self%input_max, self%norm_min, self%norm_max, sub_field_3d)
                end do
            end do

#ifdef SCOREP
            SCOREP_USER_REGION_END( normalize_fields_handle )       
#endif

        end subroutine

        
        subroutine denormalize_2D(self, x, xmin, xmax, a, b, xdenorm)
            class(ml_coupling_t) :: self 
            
            real(kind=8), dimension(:,:), intent(in)  :: x
            real(kind=8),                   intent(in)  :: xmin, xmax, a, b
            real(kind=8), dimension(:,:), intent(out) :: xdenorm

            xdenorm = xmin + ((x(:,:) - a) * (xmax- xmin)) / (b-a)
        end subroutine  
        
        subroutine denormalize_3D(self, x, xmin, xmax, a, b, xdenorm)
            class(ml_coupling_t) :: self 
            
            real(kind=8), dimension(:,:,:), intent(in)  :: x
            real(kind=8),                   intent(in)  :: xmin, xmax, a, b
            real(kind=8), dimension(:,:,:), intent(out) :: xdenorm

            xdenorm = xmin + ((x(:,:,:) - a) * (xmax- xmin)) / (b-a)
        end subroutine 

        subroutine denormalize_fields(self, output_fields_post)
            class(ml_coupling_t) :: self

            real(kind=8), dimension(:,:,:,:,:), intent(inout), target  :: output_fields_post

            integer :: b, c
            real(kind=8), dimension(:,:,:), pointer :: sub_field_3d

#ifdef SCOREP
            SCOREP_USER_REGION_DEFINE( denormalize_fields_handle )
            SCOREP_USER_REGION_BEGIN( denormalize_fields_handle, "denormalize_fields", SCOREP_USER_REGION_TYPE_COMMON)
#endif

            do b = 1, size(output_fields_post, 1)
                do c = 1, size(output_fields_post, 5)
                    sub_field_3d => output_fields_post(b,:,:,:,c)
                    call self%denormalize(sub_field_3d, self%output_min, &
                    self%output_max, self%norm_min, self%norm_max, sub_field_3d)
                end do
            end do

#ifdef SCOREP
            SCOREP_USER_REGION_END( denormalize_fields_handle )       
#endif

        end subroutine

        subroutine ml_step(self)
            class(ml_coupling_t) :: self
#ifdef SCOREP
            SCOREP_USER_REGION_DEFINE( ml_step_handle )
            SCOREP_USER_REGION_DEFINE( preprocess_handle )
            SCOREP_USER_REGION_DEFINE( inference_handle )
            SCOREP_USER_REGION_DEFINE( postprocess_handle )
#endif

#ifdef SCOREP
            SCOREP_USER_REGION_BEGIN( ml_step_handle, "ml_coupling_ml_step", SCOREP_USER_REGION_TYPE_DYNAMIC)
            SCOREP_USER_REGION_BEGIN( preprocess_handle, "ml_coupling_preprocess_input", SCOREP_USER_REGION_TYPE_COMMON)
#endif
            call self%preprocess_input(self%input_fields, self%input_fields_pre)
#ifdef SCOREP
            SCOREP_USER_REGION_END( preprocess_handle )
            SCOREP_USER_REGION_BEGIN( inference_handle, "ml_coupling_inference", SCOREP_USER_REGION_TYPE_COMMON)
#endif
            call self%inference(self%input_fields_pre, self%output_fields_post)
#ifdef SCOREP
            SCOREP_USER_REGION_END( inference_handle )
            SCOREP_USER_REGION_BEGIN( postprocess_handle, "ml_coupling_postprocess_output", SCOREP_USER_REGION_TYPE_COMMON)
#endif
            call self%postprocess_output(self%output_fields_post, self%output_fields)

#ifdef SCOREP
            SCOREP_USER_REGION_END( postprocess_handle )
            SCOREP_USER_REGION_END( ml_step_handle )
#endif

        end subroutine

end module
