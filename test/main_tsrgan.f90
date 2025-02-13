program main_tsrgan

    use mpi
    use h5fortran

    use ml_coupling_turbulence

    implicit none

    integer :: dim_x, dim_y, dim_z, n_channels
    integer, dimension(3) :: ghost_cells
    integer :: upsampling

    character(len=1) :: coupling_strategy_arg
    integer :: coupling_strategy_id
    character(len=256) :: project_root

    integer :: rank, nprocs
    integer :: error

    character(len=256) :: data_path, data_path_out
    character(len=256) :: dset_u, dset_v, dset_w
    character(len=256) :: dset_tauij
    character(len=256) :: tsrgan_model
    character(len=8) :: my_rank_str

    real(kind=8), dimension(:,:,:), allocatable, target :: u, v, w, tmp
    real(kind=8), dimension(:,:,:), allocatable, target :: u_sr, v_sr, w_sr
    real(kind=8), dimension(:,:,:), allocatable :: tau11, tau12, tau13, tau22, tau23, tau33

    real(kind=8), dimension(:,:,:,:), allocatable :: input_fields, output_fields

    integer, dimension(5) :: input_shape, output_shape
    integer :: batch_size

    ! normalization variables
    real(kind=8) input_min, input_max
    real(kind=8) norm_min, norm_max
    real(kind=8) output_min, output_max

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

    dim_x = 33
    dim_y = 33
    dim_z = 33
    n_channels = 3
    upsampling = 4

    ghost_cells = [ 2, 2, 2 ]

    data_path = trim(project_root) // "/data/FHIT_32x32x32_output.h5"
    dset_u = "flow/U"
    dset_v = "flow/V"
    dset_w = "flow/W"

    dset_tauij = "flow/tau"

    tsrgan_model = trim(project_root) // "/model/tsrgan/TSRGAN_3D_36_4X_decay_gaussian.tf"

    ! allocate memory for tmp storage
    allocate(tmp(dim_x, dim_y, dim_z), source=-13.37_8, stat=error)
    if (error /= 0) then
        write(*,*) "Could not allocate array for U!"
    endif

    ! allocate memory for U
    allocate(u(dim_x, dim_y, dim_z), source=-13.37_8, stat=error)
    if (error /= 0) then
        write(*,*) "Could not allocate array for U!"
    endif

    ! allocate memory for V
    allocate(v(dim_x, dim_y, dim_z), source=-13.37_8, stat=error)
    if (error /= 0) then
        write(*,*) "Could not allocate array for V!"
    endif

    ! allocate memory for W
    allocate(w(dim_x, dim_y, dim_z), source=-13.37_8, stat=error)
    if (error /= 0) then
        write(*,*) "Could not allocate array for W!"
    endif

    ! allocate memory for tau11
    allocate(tau11(dim_x, dim_y, dim_z), source=-13.37_8, stat=error)
    if (error /= 0) then
        write(*,*) "Could not allocate array for tau11!"
    endif

    ! allocate memory for tau12
    allocate(tau12(dim_x, dim_y, dim_z), source=-13.37_8, stat=error)
    if (error /= 0) then
        write(*,*) "Could not allocate array for tau12!"
    endif

    ! allocate memory for tau13
    allocate(tau13(dim_x, dim_y, dim_z), source=-13.37_8, stat=error)
    if (error /= 0) then
        write(*,*) "Could not allocate array for tau13!"
    endif

    ! allocate memory for tau22
    allocate(tau22(dim_x, dim_y, dim_z), source=-13.37_8, stat=error)
    if (error /= 0) then
        write(*,*) "Could not allocate array for tau22!"
    endif

    ! allocate memory for tau23
    allocate(tau23(dim_x, dim_y, dim_z), source=-13.37_8, stat=error)
    if (error /= 0) then
        write(*,*) "Could not allocate array for tau23!"
    endif

    ! allocate memory for tau33
    allocate(tau33(dim_x, dim_y, dim_z), source=-13.37_8, stat=error)
    if (error /= 0) then
        write(*,*) "Could not allocate array for tau33!"
    endif

    ! allocate memory for U_SR
    allocate(u_sr(dim_x * upsampling, dim_y * upsampling, dim_z * upsampling), source=-13.37_8, stat=error)
    if (error /= 0) then
        write(*,*) "Could not allocate array for U_SR!"
    endif

    ! allocate memory for V_SR
    allocate(v_sr(dim_x * upsampling, dim_y * upsampling, dim_z * upsampling), source=-13.37_8, stat=error)
    if (error /= 0) then
        write(*,*) "Could not allocate array for V_SR!"
    endif

    ! allocate memory for W_SR
    allocate(w_sr(dim_x * upsampling, dim_y * upsampling, dim_z * upsampling), source=-13.37_8, stat=error)
    if (error /= 0) then
        write(*,*) "Could not allocate array for W_SR!"
    endif

    ! allocate memory for input_fields aka ML input tensor
    allocate(input_fields(                &
                dim_x + 2*ghost_cells(1), &
                dim_y + 2*ghost_cells(2), &
                dim_z + 2*ghost_cells(3), &
                n_channels               &
            ), stat=error, source=-13.37_8)
    if (error /= 0) then
        write(*,*) "ERROR: Could not allocate input_fields!"
    endif

    allocate(output_fields( &
                dim_x + 2*ghost_cells(1), &
                dim_y + 2*ghost_cells(2), &
                dim_z + 2*ghost_cells(3), &
                6 &
            ), stat=error, source=-42.24_8)
    if (error /= 0) then
        write(*,*) "ERROR: Could not allocate output_fields!"
    endif

    call h5read(data_path, dset_u, u)
    call h5read(data_path, dset_v, v)
    call h5read(data_path, dset_w, w)

    ! set up input fields with U,V,W data
    input_fields( &
        1 + ghost_cells(1) : dim_x + ghost_cells(1), &
        1 + ghost_cells(2) : dim_y + ghost_cells(2), &
        1 + ghost_cells(3) : dim_z + ghost_cells(3), &
        1 &
    ) = u(:,:,:)

    input_fields( &
        1 + ghost_cells(1) : dim_x + ghost_cells(1), &
        1 + ghost_cells(2) : dim_y + ghost_cells(2), &
        1 + ghost_cells(3) : dim_z + ghost_cells(3), &
        2 &
    ) = v(:,:,:)

    input_fields( &
        1 + ghost_cells(1) : dim_x + ghost_cells(1), &
        1 + ghost_cells(2) : dim_y + ghost_cells(2), &
        1 + ghost_cells(3) : dim_z + ghost_cells(3), &
        3 &
    ) = w(:,:,:)

    batch_size = 1
    input_shape = [ 1, dim_x, dim_y, dim_z, n_channels ]
    output_shape = [ 1, dim_x*upsampling, dim_y*upsampling, dim_z*upsampling, n_channels ]

    ! set variables for normalization
    
    input_min = -90.0
    input_max = 90.0 ! magic number taken from NormalizationValues.txt
    norm_min = 0
    norm_max = 1
    output_min = -90.0 
    output_max = 90.0 ! magic number taken from NormalizationValues.txt

    allocate(ml_coupling_turbulence_t :: ml_coupler)
    call ml_coupler%init(input_fields, output_fields, tsrgan_model, batch_size, input_shape, output_shape, ghost_cells, input_min, input_max, norm_min, norm_max, output_min, output_max, coupling_strategy_id, MPI_COMM_WORLD)

    call ml_coupler%ml_step()

    tau11(:,:,:) = output_fields( &
                        1 + ghost_cells(1) : dim_x + ghost_cells(1), &
                        1 + ghost_cells(2) : dim_y + ghost_cells(2), &
                        1 + ghost_cells(3) : dim_z + ghost_cells(3), &
                        1 &
                    )

    tau12(:,:,:) = output_fields( &
                        1 + ghost_cells(1) : dim_x + ghost_cells(1), &
                        1 + ghost_cells(2) : dim_y + ghost_cells(2), &
                        1 + ghost_cells(3) : dim_z + ghost_cells(3), &
                        2 &
                    )

    tau13(:,:,:) = output_fields( &
                        1 + ghost_cells(1) : dim_x + ghost_cells(1), &
                        1 + ghost_cells(2) : dim_y + ghost_cells(2), &
                        1 + ghost_cells(3) : dim_z + ghost_cells(3), &
                        3 &
                    )

    tau22(:,:,:) = output_fields( &
                        1 + ghost_cells(1) : dim_x + ghost_cells(1), &
                        1 + ghost_cells(2) : dim_y + ghost_cells(2), &
                        1 + ghost_cells(3) : dim_z + ghost_cells(3), &
                        4 &
                    )

    tau23(:,:,:) = output_fields( &
                        1 + ghost_cells(1) : dim_x + ghost_cells(1), &
                        1 + ghost_cells(2) : dim_y + ghost_cells(2), &
                        1 + ghost_cells(3) : dim_z + ghost_cells(3), &
                        5 &
                    )

    tau33(:,:,:) = output_fields( &
                        1 + ghost_cells(1) : dim_x + ghost_cells(1), &
                        1 + ghost_cells(2) : dim_y + ghost_cells(2), &
                        1 + ghost_cells(3) : dim_z + ghost_cells(3), &
                        6 &
                    )

    ! write results to file
    write(my_rank_str, '(I0)') rank
    if (coupling_strategy_id == 1) then
        data_path_out = trim(project_root) // "/data/FHIT_128x128x128_" // trim(adjustl(my_rank_str)) // "_inferred_aix.h5"
    end if

    if (coupling_strategy_id == 2) then
        data_path_out = trim(project_root) // "/data/FHIT_128x128x128_" // trim(adjustl(my_rank_str)) // "_inferred_phydll.h5"
    end if
    
    write(*,*) "data_path_out = ", data_path_out

    call h5write(data_path_out, trim(dset_tauij)//"_11", tau11)
    call h5write(data_path_out, trim(dset_tauij)//"_12", tau12)
    call h5write(data_path_out, trim(dset_tauij)//"_13", tau13)
    call h5write(data_path_out, trim(dset_tauij)//"_22", tau22)
    call h5write(data_path_out, trim(dset_tauij)//"_23", tau23)
    call h5write(data_path_out, trim(dset_tauij)//"_33", tau33)

    ! finalize ML coupling
    call ml_coupler%finalize()

    call MPI_Finalize(error)

    write(*,*) "Program successfully closed."

end program main_tsrgan