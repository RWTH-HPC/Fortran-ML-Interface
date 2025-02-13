program main_cnn_test_2D

    use mpi
    use ml_coupling
    use ml_coupling_combustion

    implicit none

    character(len=256) :: data_path, model_path

    integer :: error
    integer :: dim_x, dim_y, dim_z
    integer :: n_ch ! number of channels, e.g 1 = scalar, 3 = [U,V,W]
    integer, dimension(3) :: ghost_cells

    real(kind=8), dimension(:,:,:,:), allocatable :: input_fields
    real(kind=8), dimension(:,:,:,:), allocatable :: output_fields

    real(kind=8), dimension(:,:), allocatable :: ml_input, ml_output
    real(kind=8), dimension(:,:,:), allocatable, target :: ml_input_3D, ml_output_3D

    class(ml_coupling_t), pointer :: ml_coupler
    integer :: batch_size
    integer, dimension(5) :: input_shape, output_shape
    integer :: coupling_strategy_id
    character(len=1) :: coupling_strategy_arg
    character(len=256) :: project_root

    real(kind=8) :: input_min, input_max
    real(kind=8) :: norm_min, norm_max
    real(kind=8) :: output_min, output_max


    if (command_argument_count() .lt. 1) then
        write(*,*) "ERROR: No command line arguments specified! Usage ./main.x <coupling_strategy_id>."
        stop
    else
        call get_command_argument(1, coupling_strategy_arg)
    endif

    call get_environment_variable("FORTRAN_ML_ROOT", project_root, status=error)
    if(error /= 0) then
        write(*,*) "Error: Please set the environment variable FORTRAN_ML_ROOT to the root directory of the FORTRAN-ML-INTERFACE project."
        stop error
    endif

    call MPI_Init(error)

    model_path = trim(project_root) // "/model/cnn2d-test/testConvolution2D.tf"

    dim_x = 3
    dim_y = 3
    dim_z = 1
    n_ch = 1

    ghost_cells = [ 2, 2, 2 ]

    ! normalization parameters
    ! if everything is [0,1] there should efectively be no normalization
    input_min = 0
    input_max = 1
    norm_min = 0
    norm_max = 1
    output_min = 0 
    output_max = 1

    allocate(input_fields(  dim_x + 2*ghost_cells(1), &
                            dim_y + 2*ghost_cells(2), &
                            dim_z + 2*ghost_cells(3), &
                            n_ch                      &
                         ), source=-13.37_8, stat=error)
    if (error /= 0) then
        write(*,*) "ERROR: Could not allocate input_fields!"
    endif

    allocate(output_fields( dim_x - 1 + 2*ghost_cells(1), &
                            dim_y - 1 + 2*ghost_cells(2), &
                            dim_z     + 2*ghost_cells(2), &
                            n_ch                          &
                         ), source=-42.24_8, stat=error)
    if (error /= 0) then
        write(*,*) "ERROR: Could not allocate output_fields!"
    endif

    allocate(ml_input(dim_x, dim_y), stat=error)
    if (error /= 0) then
        write(*,*) "ERROR: Could not allocate ml_input!"
    endif

    allocate(ml_input_3D(   dim_x + 2*ghost_cells(1), &
                            dim_y + 2*ghost_cells(2), &
                            dim_z + 2*ghost_cells(3)  &
                        ), source=-13.37_8, stat=error)
    if (error /= 0) then
        write(*,*) "ERROR: Could not allocate ml_input_3D!"
    endif

    allocate(ml_output(dim_x - 1, dim_y - 1), stat=error)
    if (error /= 0) then
        write(*,*) "ERROR: Could not allocate ml_output!"
    endif

    allocate(ml_output_3D(  dim_x - 1 + 2*ghost_cells(1), &
                            dim_y - 1 + 2*ghost_cells(2), &
                            dim_z     + 2*ghost_cells(2)  &
                         ), source=-42.24_8, stat=error)
    if (error /= 0) then
        write(*,*) "ERROR: Could not allocate ml_output_3D!"
    endif

    allocate(ml_coupling_combustion_t :: ml_coupler)
    batch_size = 1
    input_shape = [1, dim_x, dim_y, 1, 1]
    output_shape = [1, dim_x - 1, dim_y - 1, 1, 1]

    ! fill input
    ml_input(:,1) = [1, 2, 3]
    ml_input(:,2) = [4, 5, 6]
    ml_input(:,3) = [7, 8, 9]
    ml_input_3D(    1 + ghost_cells(1) : dim_x + ghost_cells(1), &
                    1 + ghost_cells(2) : dim_y + ghost_cells(2), &
                    ceiling(size(ml_input_3D, 3) / 2.0_8)       &
                ) = ml_input(:,:)

    ! associate input and output fields
    input_fields(:,:,:, n_ch) = ml_input_3D(:,:,:)

    ! convert coupling_strategy_id from command line argument string to integer
    read(coupling_strategy_arg, '(I1)') coupling_strategy_id
    call ml_coupler%init(input_fields, output_fields, model_path, batch_size, input_shape, output_shape, ghost_cells, input_min, input_max, norm_min, norm_max, output_min, output_max, coupling_strategy_id, MPI_COMM_WORLD)

    call ml_coupler%ml_step()

    ! extract output
    ml_output_3D(:,:,:) = output_fields(:,:,:, n_ch)
    ml_output(:,:) = ml_output_3D( 1 + ghost_cells(1) : dim_x - 1 + ghost_cells(1), &
                                   1 + ghost_cells(2) : dim_x - 1 + ghost_cells(2), &
                                   ceiling(size(ml_output_3D, 3) / 2.0_8)           &
                                 )
    
    write(*,*) "Output:"
    write(*,*) "Row 1 = ", ml_output(:,1)
    write(*,*) "Row 2 = ", ml_output(:,2)

    call ml_coupler%finalize()


    call MPI_Finalize(error)

end program
