program main_cnn_test_3D_multichannel

    use mpi
    use ml_coupling
    use ml_coupling_combustion

    implicit none

    character(len=256) :: data_path, model_path

    integer :: error
    integer :: dim_x, dim_y, dim_z
    integer :: n_ch ! number of channels, e.g 1 = scalar, 3 = [U,V,W]
    integer, dimension(3) :: ghost_cells 

    real(kind=8), dimension(:,:,:,:), allocatable :: input_fields, output_fields
    real(kind=8), dimension(:,:,:), allocatable, target :: input_field1, input_field2, input_field3
    real(kind=8), dimension(:,:,:), allocatable, target :: output_field  

    real(kind=8), dimension(:,:,:), allocatable :: ml_input1, ml_input2, ml_input3
    real(kind=8), dimension(:,:,:), allocatable :: ml_output

    class(ml_coupling_t), pointer :: ml_coupler
    integer :: batch_size
    integer, dimension(5) :: input_shape, output_shape
    integer :: coupling_strategy_id
    character(len=1) :: coupling_strategy_arg
    character(len=256) :: project_root

    real(kind=8) :: input_min, input_max
    real(kind=8) :: norm_min, norm_max
    real(kind=8) :: output_min, output_max

    integer :: rank


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
    call MPI_Comm_rank(MPI_COMM_WORLD, rank, error)

    model_path = trim(project_root) // "/model/cnn3d-multichannel-test/testConvolution3D-multichannel.tf"

    dim_x = 3
    dim_y = 3
    dim_z = 3
    n_ch = 3
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
                         ), stat=error, source=-13.37_8)
    if (error /= 0) then
        write(*,*) "ERROR: Could not allocate input_fields!"
    endif

    allocate(input_field1(  dim_x + 2*ghost_cells(1), &
                            dim_y + 2*ghost_cells(2), &
                            dim_z + 2*ghost_cells(3)  &
                        ), stat=error, source=-13.37_8)
    if (error /= 0) then
        write(*,*) "ERROR: Could not allocate input_field1!"
    endif

    allocate(input_field2(  dim_x + 2*ghost_cells(1), &
                            dim_y + 2*ghost_cells(2), &
                            dim_z + 2*ghost_cells(3)  &
                        ), stat=error, source=-13.37_8)
    if (error /= 0) then
        write(*,*) "ERROR: Could not allocate input_field2!"
    endif

    allocate(input_field3(  dim_x + 2*ghost_cells(1), &
                            dim_y + 2*ghost_cells(2), &
                            dim_z + 2*ghost_cells(3)  &
                        ), stat=error, source=-13.37_8)
    if (error /= 0) then
        write(*,*) "ERROR: Could not allocate input_field3!"
    endif

    allocate(output_fields( dim_x - 1 + 2*ghost_cells(1), &
                            dim_y - 1 + 2*ghost_cells(2), &
                            dim_z - 1 + 2*ghost_cells(3), &
                            1                             &
                          ), stat=error, source=-42.24_8)
    if (error /= 0) then
        write(*,*) "ERROR: Could not allocate output_fields!"
    endif

    allocate(output_field(  dim_x - 1 + 2*ghost_cells(1), &
                            dim_y - 1 + 2*ghost_cells(2), &
                            dim_z - 1 + 2*ghost_cells(3)  &
                         ), stat=error, source=-42.24_8)
    if (error /= 0) then
        write(*,*) "ERROR: Could not allocate output_field!"
    endif

    allocate(ml_input1(dim_x, dim_y, dim_z), stat=error)
    if (error /= 0) then
        write(*,*) "ERROR: Could not allocate ml_input1!"
    endif
    allocate(ml_input2(dim_x, dim_y, dim_z), stat=error)
    if (error /= 0) then
        write(*,*) "ERROR: Could not allocate ml_input2!"
    endif
    allocate(ml_input3(dim_x, dim_y, dim_z), stat=error)
    if (error /= 0) then
        write(*,*) "ERROR: Could not allocate ml_input3!"
    endif

    allocate(ml_output(dim_x - 1, dim_y - 1, dim_z - 1), stat=error)
    if (error /= 0) then
        write(*,*) "ERROR: Could not allocate ml_output!"
    endif

    allocate(ml_coupling_combustion_t :: ml_coupler)
    batch_size = 1
    input_shape = [1, dim_x, dim_y, dim_z, n_ch]
    output_shape = [1, dim_x - 1, dim_y - 1, dim_z - 1, 1]

    ! fill input
    ! channel 1
    ml_input1(:,1,1) = [ 1,  2,  3]
    ml_input1(:,2,1) = [ 4,  5,  6]
    ml_input1(:,3,1) = [ 7,  8,  9]

    ml_input1(:,1,2) = [10, 11, 12]
    ml_input1(:,2,2) = [13, 14, 15]
    ml_input1(:,3,2) = [16, 17, 18]

    ml_input1(:,1,3) = [19, 20, 21]
    ml_input1(:,2,3) = [22, 23, 24]
    ml_input1(:,3,3) = [25, 26, 27]

    ml_input1(:,:,:) = ml_input1(:,:,:) * (rank+1)

    input_field1(   1 + ghost_cells(1) : dim_x + ghost_cells(1), &
                    1 + ghost_cells(2) : dim_y + ghost_cells(2), &
                    1 + ghost_cells(3) : dim_z + ghost_cells(3)  &
                ) = ml_input1(:,:,:)

    ! channel 2
    ml_input2(:,1,1) = [28, 29, 30]
    ml_input2(:,2,1) = [31, 32, 33]
    ml_input2(:,3,1) = [34, 35, 36]

    ml_input2(:,1,2) = [37, 38, 39]
    ml_input2(:,2,2) = [40, 41, 42]
    ml_input2(:,3,2) = [43, 44, 45]

    ml_input2(:,1,3) = [46, 47, 48]
    ml_input2(:,2,3) = [49, 50, 51]
    ml_input2(:,3,3) = [52, 53, 54] 

    ml_input2(:,:,:) = ml_input2(:,:,:) * (rank+1)

    input_field2(   1 + ghost_cells(1) : dim_x + ghost_cells(1), &
                    1 + ghost_cells(2) : dim_y + ghost_cells(2), &
                    1 + ghost_cells(3) : dim_z + ghost_cells(3)  &
                ) = ml_input2(:,:,:)

    ! channel 3
    ml_input3(:,1,1) = [55, 56, 57]
    ml_input3(:,2,1) = [58, 59, 60]
    ml_input3(:,3,1) = [61, 62, 63]

    ml_input3(:,1,2) = [64, 65, 66]
    ml_input3(:,2,2) = [67, 68, 69]
    ml_input3(:,3,2) = [70, 71, 72]

    ml_input3(:,1,3) = [73, 74, 75]
    ml_input3(:,2,3) = [76, 77, 78]
    ml_input3(:,3,3) = [79, 80, 81] 

    ml_input3(:,:,:) = ml_input3(:,:,:) * (rank+1)

    input_field3(   1 + ghost_cells(1) : dim_x + ghost_cells(1), &
                    1 + ghost_cells(2) : dim_y + ghost_cells(2), &
                    1 + ghost_cells(3) : dim_z + ghost_cells(3)  &
                ) = ml_input3(:,:,:)

    ! associate input and output fields
    input_fields(:,:,:,1) = input_field1(:,:,:)
    input_fields(:,:,:,2) = input_field2(:,:,:)
    input_fields(:,:,:,3) = input_field3(:,:,:)

    ! convert coupling_strategy_id from command line argument string to integer
    read(coupling_strategy_arg, '(I1)') coupling_strategy_id
    call ml_coupler%init(input_fields, output_fields, model_path, batch_size, input_shape, output_shape, ghost_cells, input_min, input_max, norm_min, norm_max, output_min, output_max, coupling_strategy_id, MPI_COMM_WORLD)

    call ml_coupler%ml_step()

    ! extract output
    output_field(:,:,:) = output_fields(:,:,:,1)
    ml_output(:,:,:) = output_field(1 + ghost_cells(1) : dim_x - 1 + ghost_cells(1), &
                                    1 + ghost_cells(2) : dim_x - 1 + ghost_cells(2), &
                                    1 + ghost_cells(3) : dim_z - 1 + ghost_cells(3)  &
                                   )
    
    write(*,*) "Output:"
    write(*,*) "Matrix 1:"
    write(*,*) "Row 1 = ", ml_output(:,1,1)
    write(*,*) "Row 2 = ", ml_output(:,2,1)

    write(*,*) "Matrix 2:"
    write(*,*) "Row 1 = ", ml_output(:,1,2)
    write(*,*) "Row 2 = ", ml_output(:,2,2)

    call ml_coupler%finalize()

    call MPI_Finalize(error)

    write(*,*) "Finalize done!"
    stop

end program
