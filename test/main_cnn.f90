program main_cnn

    use mpi
    use h5fortran
    use ml_coupling_combustion

    implicit none

    integer :: dim_x, dim_y, dim_z, n_channels

    real(kind=8), dimension(:,:,:), allocatable, target :: c ! progress variable (H2O)
    real(kind=8), dimension(:,:,:), allocatable, target :: omega ! reaction rate (of H2O)
    real(kind=8), dimension(:,:,:), allocatable, target :: omega_pred ! reaction rate (of H2O) predicted by UNet

    !type(field_ptr), dimension(:), allocatable :: input_fields, output_fields
    real(kind=8), dimension(:,:,:,:), allocatable :: input_fields, output_fields
    
    integer, dimension(5) :: input_shape, output_shape
    integer, dimension(3) :: ghost_cells

    ! normalization variables
    real(kind=8) input_min, input_max
    real(kind=8) norm_min, norm_max
    real(kind=8) output_min, output_max

    integer :: coupling_strategy_id
    character(len=1) :: coupling_strategy_arg
    integer :: batch_size

    character(len=256) :: project_root
    character(len=256) :: data_path_base, data_path, omega_model, data_path_out
    character(len=256) :: dset_c, dset_omega, dset_omega_pred
    character(len=8) :: my_rank_str

    integer :: rank, nprocs
    integer :: error

    real(kind=8), dimension(:,:,:), pointer :: field
    real(kind=8), dimension(:,:,:), allocatable, target :: c_3D, omega_3D

    class(ml_coupling_t), pointer :: ml_coupler


    if (command_argument_count() .lt. 1) then
        write(*,*) "ERROR: No command line arguments specified! Usage ./main.x <coupling_strategy_id>."
        stop
    else
        call get_command_argument(1, coupling_strategy_arg)
        ! convert coupling_strategy_id from command line argument string to integer
        read(coupling_strategy_arg, '(I1)') coupling_strategy_id
    endif

    call get_environment_variable("FORTRAN_ML_ROOT", project_root, status=error)
    if(error /= 0) then
        write(*,*) "Error: Please set the environment variable FORTRAN_ML_ROOT to the root directory of the FORTRAN-ML-INTERFACE project."
        stop error
    endif
 
    call MPI_Init(error)
    call MPI_Comm_rank(MPI_COMM_WORLD, rank, error)
    call MPI_Comm_size(MPI_COMM_WORLD, nprocs, error)

    dim_x = 2048
    dim_y = 2048    
    dim_z = 5
    n_channels = 1

    ghost_cells = [ 2, 2, 2]

    data_path = trim(project_root) // "/data/H2_000057_prate.h5"
    write(my_rank_str, '(I0)') rank
    data_path_out = trim(project_root) // "/data/H2_000057_" // trim(adjustl(my_rank_str)) // "_prate_inferred.h5"
    write(*,*) "data path out = ", data_path_out

    dset_c = "sd_1d_test/data/cv_data_real/H2O"
    dset_omega = "sd_1d_test/data/cv_data_real/source_H2O"
    dset_omega_pred = "sd_1d_test/data/cv_data_real/omega"

    omega_model = trim(project_root) // "/model/unet/_2D/Model_1251_loss_0.00012877.tf"

    ! allocate memory for progress variable c
    allocate(c(dim_x, dim_y, 1), source=-13.37_8, stat=error)
    if (error /= 0) then
        write(*,*) "Could not allocate array for progress variable c!"
    endif

    ! allocate memory for true reaction rate omega
    allocate(omega(dim_x, dim_y, 1), source=-13.37_8, stat=error)
    if (error /= 0) then
        write(*,*) "Could not allocate array for TRUE reaction rate omega!"
    endif

     ! allocate memory for predicted reaction rate omega
    allocate(omega_pred(dim_x, dim_y, 1), source=-13.37_8, stat=error)
    if (error /= 0) then
        write(*,*) "Could not allocate array for PRED reaction rate omega!"
    endif

    allocate(input_fields(dim_x, dim_y, dim_z, n_channels), source=-13.37_8, stat=error)
    if (error /= 0) then
        write(*,*) "Could not allocate array for input fields!"
    endif

    allocate(c_3D(dim_x, dim_y, dim_z), source=-13.37_8, stat=error)
    if (error /= 0) then
        write(*,*) "Could not allocate array for c_3D!"
    endif 

    allocate(omega_3D(dim_x, dim_y, dim_z), source=-13.37_8, stat=error)
    if (error /= 0) then
        write(*,*) "Could not allocate array for omega_3D!"
    endif 

    allocate(output_fields(dim_x, dim_y, dim_z, n_channels), source=-13.37_8, stat=error)
    if (error /= 0) then
        write(*,*) "Could not allocate array for output fields!"
    endif

    ! read progress variable c from h5 file
    call h5read(data_path, dset_c, c)
    c_3D(:,:,ceiling(dim_z / 2.0_8)) = c(:,:,1)
    
    ! associate input and output fields
    input_fields(:,:,:,n_channels) = c_3D(:,:,:)

    ! read reaction rate omega from h5 file
    call h5read(data_path, dset_omega, omega)

    input_shape =  [ 1, dim_x - 2*ghost_cells(1), dim_y - 2*ghost_cells(2) , 1, 1 ]
    output_shape = [ 1, dim_x - 2*ghost_cells(1), dim_y - 2*ghost_cells(2) , 1, 1 ]
    batch_size = 1

    ! set variables for normalization
    ! for NCSA-MLLIB normalization/denormalization is done in Python!
    
    input_min = 0
    input_max = 0.1242 ! magic number taken from mllib.py
    norm_min = -1
    norm_max = 1
    output_min = 0 
    output_max = 221.5949 ! magic number taken from mllib.py

    ! init ML coupling
    allocate(ml_coupling_combustion_t :: ml_coupler)
    call ml_coupler%init(input_fields, output_fields, omega_model, batch_size, input_shape, output_shape, ghost_cells, input_min, input_max, norm_min, norm_max, output_min, output_max, coupling_strategy_id, MPI_COMM_WORLD)

    ! let the coupler perform an ML step; input and output have been set during init
    call ml_coupler%ml_step()
    write(*,*) "ML STEP() DONE!"

    omega_3D(:,:,:) = output_fields(:,:,:, n_channels)
    omega_pred(:,:,1) = omega_3D(:,:,ceiling(dim_z / 2.0_8))
    write(*,*) "shape of omega_pred = ", shape(omega_pred)
    write(*,*) "shape of output field = ", shape(omega_3D(:,:,ceiling(dim_z / 2.0_8)))
    ! write results to file
    call h5write(data_path_out, dset_c, c)
    call h5write(data_path_out, dset_omega, omega)
    call h5write(data_path_out, dset_omega_pred, omega_pred)

    ! finalize ML coupling
    call ml_coupler%finalize()

    call MPI_Finalize(error)

    write(*,*) "Program successfully closed."

end program