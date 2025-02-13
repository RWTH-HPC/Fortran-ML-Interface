# main_cnn executable with mlCouplingLib
add_executable(main_cnn_lib.x main_cnn.f90)

target_include_directories(
    main_cnn_lib.x
        PRIVATE
        ${CMAKE_SOURCE_DIR}/extern/h5fortran/BUILD/INSTALL/include
)

target_link_libraries(main_cnn_lib.x PRIVATE h5fortran::h5fortran)
target_link_libraries(main_cnn_lib.x PRIVATE mlCouplingLib)
target_link_libraries(main_cnn_lib.x PRIVATE ${TENSORFLOW_LIBRARIES})
if(WITH_AIX)
    target_link_libraries(main_cnn_lib.x PRIVATE ${AIX_BUILD_DIR}/lib/libAIxeleratorService.so)
endif()
if(WITH_PHYDLL)
    target_link_libraries(main_cnn_lib.x PRIVATE ${PHYDLL_BUILD_DIR}/lib/libphydll_f.so)
    target_link_libraries(main_cnn_lib.x PRIVATE ${PHYDLL_BUILD_DIR}/lib/libphydll.so)   
endif()

install(
    TARGETS main_cnn_lib.x
    RUNTIME DESTINATION ${CMAKE_BINARY_DIR}/bin
)

if(WITH_SCOREP)
    scorep_instrument_target(
        main_cnn_lib.x
            USER ON
            COMPILER ON
    )
endif()


# main_cnn executable
#if(1 AND 0)
add_executable(main_cnn.x main_cnn.f90)

target_sources(
    main_cnn.x
    PRIVATE
        main_cnn.f90
        ${CMAKE_SOURCE_DIR}/src/ml_coupling/ml_coupling.F90
        ${CMAKE_SOURCE_DIR}/src/ml_coupling/ml_coupling_combustion.F90
        ${CMAKE_SOURCE_DIR}/src/ml_coupling_strategy/ml_coupling_strategy.f90
)

target_include_directories(main_cnn.x PRIVATE ${MPI_Fortran_INCLUDE_DIRS})

target_link_libraries(main_cnn.x PRIVATE h5fortran::h5fortran)
target_link_libraries(main_cnn.x PRIVATE ${MPI_LIBRARIES})

if(WITH_AIX)
    target_sources(
        main_cnn.x
        PRIVATE
            ${CMAKE_SOURCE_DIR}/src/ml_coupling_strategy/aix/ml_coupling_strategy_aix.f90
    )
    # aixelerator_service_mod comes from here (maybe find a better place in the AIxeleratorService project for it)
    target_include_directories(main_cnn.x PRIVATE ${AIX_BUILD_DIR}/test)
    target_link_libraries(main_cnn.x PRIVATE ${AIX_BUILD_DIR}/lib/libAIxeleratorService.so)
    target_link_libraries(main_cnn.x PRIVATE ${TENSORFLOW_LIBRARIES})
endif()
if(WITH_PHYDLL)
    target_sources(
        main_cnn.x
        PRIVATE
            ${CMAKE_SOURCE_DIR}/src/ml_coupling_strategy/phydll/ml_coupling_strategy_phydll.F90
    )
    target_include_directories(main_cnn.x PRIVATE ${PHYDLL_BUILD_DIR}/include)
    target_link_libraries(main_cnn.x PRIVATE ${PHYDLL_BUILD_DIR}/lib/libphydll_f.so)
    target_link_libraries(main_cnn.x PRIVATE ${PHYDLL_BUILD_DIR}/lib/libphydll.so)
endif()

install(
    TARGETS main_cnn.x
    RUNTIME DESTINATION ${CMAKE_BINARY_DIR}/bin
)

if(WITH_SCOREP)
    scorep_instrument_target(
        main_cnn.x
            USER ON
            COMPILER ON
    )
endif()
#endif()


# main_cnn_3D executable
if(1 AND 0)
add_executable(main_cnn_3D.x main_cnn_3D.f90)

target_sources(
    main_cnn_3D.x
    PRIVATE
        main_cnn_3D.f90
        ${CMAKE_SOURCE_DIR}/src/ml_coupling/ml_coupling.F90
        ${CMAKE_SOURCE_DIR}/src/ml_coupling/ml_coupling_combustion.F90
        ${CMAKE_SOURCE_DIR}/src/ml_coupling_strategy/ml_coupling_strategy.f90
)

target_include_directories(main_cnn_3D.x PRIVATE ${MPI_Fortran_INCLUDE_DIRS})

target_link_libraries(main_cnn_3D.x PRIVATE h5fortran::h5fortran)
target_link_libraries(main_cnn_3D.x PRIVATE ${MPI_LIBRARIES})

if(WITH_AIX)
    target_sources(
        main_cnn_3D.x
        PRIVATE
            ${CMAKE_SOURCE_DIR}/src/ml_coupling_strategy/aix/ml_coupling_strategy_aix.f90
    )
    # aixelerator_service_mod comes from here (maybe find a better place in the AIxeleratorService project for it)
    target_include_directories(main_cnn_3D.x PRIVATE ${AIX_BUILD_DIR}/test)
    target_link_libraries(main_cnn_3D.x PRIVATE ${AIX_BUILD_DIR}/lib/libAIxeleratorService.so)
    target_link_libraries(main_cnn_3D.x PRIVATE ${TENSORFLOW_LIBRARIES})
endif()
if(WITH_PHYDLL)
    target_sources(
        main_cnn_3D.x
        PRIVATE
            ${CMAKE_SOURCE_DIR}/src/ml_coupling_strategy/phydll/ml_coupling_strategy_phydll.F90
    )
    target_include_directories(main_cnn_3D.x PRIVATE ${PHYDLL_BUILD_DIR}/include)
    target_link_libraries(main_cnn_3D.x PRIVATE ${PHYDLL_BUILD_DIR}/lib/libphydll_f.so)
    target_link_libraries(main_cnn_3D.x PRIVATE ${PHYDLL_BUILD_DIR}/lib/libphydll.so)
endif()

install(
    TARGETS main_cnn_3D.x
    RUNTIME DESTINATION ${CMAKE_BINARY_DIR}/bin
)

if(WITH_SCOREP)
    scorep_instrument_target(
        main_cnn_3D.x
            USER ON
            COMPILER ON
    )
endif()
endif()

# Test 2D Convoultion Executable
add_executable(main_cnn_test_2D.x main_cnn_test_2D.f90)

target_sources(
    main_cnn_test_2D.x
    PRIVATE
        main_cnn_test_2D.f90
        ${CMAKE_SOURCE_DIR}/src/ml_coupling/ml_coupling.F90
        ${CMAKE_SOURCE_DIR}/src/ml_coupling/ml_coupling_combustion.F90
        ${CMAKE_SOURCE_DIR}/src/ml_coupling_strategy/ml_coupling_strategy.f90
)

target_include_directories(main_cnn_test_2D.x PRIVATE ${MPI_Fortran_INCLUDE_DIRS})
target_link_libraries(main_cnn_test_2D.x PRIVATE ${MPI_LIBRARIES})

if(WITH_AIX)
    target_sources(
        main_cnn_test_2D.x
        PRIVATE
            ${CMAKE_SOURCE_DIR}/src/ml_coupling_strategy/aix/ml_coupling_strategy_aix.f90
    )
    # aixelerator_service_mod comes from here (maybe find a better place in the AIxeleratorService project for it)
    target_include_directories(main_cnn_test_2D.x PRIVATE ${AIX_BUILD_DIR}/test)
    target_link_libraries(main_cnn_test_2D.x PRIVATE ${AIX_BUILD_DIR}/lib/libAIxeleratorService.so)
    target_link_libraries(main_cnn_test_2D.x PRIVATE ${TENSORFLOW_LIBRARIES})
endif()
if(WITH_PHYDLL)
    target_sources(
        main_cnn_test_2D.x
        PRIVATE
            ${CMAKE_SOURCE_DIR}/src/ml_coupling_strategy/phydll/ml_coupling_strategy_phydll.F90
    )
    target_include_directories(main_cnn_test_2D.x PRIVATE ${PHYDLL_BUILD_DIR}/include)
    target_link_libraries(main_cnn_test_2D.x PRIVATE ${PHYDLL_BUILD_DIR}/lib/libphydll_f.so)
    target_link_libraries(main_cnn_test_2D.x PRIVATE ${PHYDLL_BUILD_DIR}/lib/libphydll.so)
endif()

install(
    TARGETS main_cnn_test_2D.x
    RUNTIME DESTINATION ${CMAKE_BINARY_DIR}/bin
)

if(WITH_SCOREP)
    scorep_instrument_target(
        main_cnn_test_2D.x
            USER ON
            COMPILER ON
    )
endif()



# Test 3D Convoultion Executable
add_executable(main_cnn_test_3D.x main_cnn_test_3D.f90)

target_sources(
    main_cnn_test_3D.x
    PRIVATE
    main_cnn_test_3D.f90
        ${CMAKE_SOURCE_DIR}/src/ml_coupling/ml_coupling.F90
        ${CMAKE_SOURCE_DIR}/src/ml_coupling/ml_coupling_combustion.F90
        ${CMAKE_SOURCE_DIR}/src/ml_coupling_strategy/ml_coupling_strategy.f90
)

target_include_directories(main_cnn_test_3D.x PRIVATE ${MPI_Fortran_INCLUDE_DIRS})
target_link_libraries(main_cnn_test_3D.x PRIVATE ${MPI_LIBRARIES})

if(WITH_AIX)
    target_sources(
        main_cnn_test_3D.x
        PRIVATE
            ${CMAKE_SOURCE_DIR}/src/ml_coupling_strategy/aix/ml_coupling_strategy_aix.f90
    )
    # aixelerator_service_mod comes from here (maybe find a better place in the AIxeleratorService project for it)
    target_include_directories(main_cnn_test_3D.x PRIVATE ${AIX_BUILD_DIR}/test)
    target_link_libraries(main_cnn_test_3D.x PRIVATE ${AIX_BUILD_DIR}/lib/libAIxeleratorService.so)
    target_link_libraries(main_cnn_test_3D.x PRIVATE ${TENSORFLOW_LIBRARIES})
endif()
if(WITH_PHYDLL)
    target_sources(
        main_cnn_test_3D.x
        PRIVATE
            ${CMAKE_SOURCE_DIR}/src/ml_coupling_strategy/phydll/ml_coupling_strategy_phydll.F90
    )
    target_include_directories(main_cnn_test_3D.x PRIVATE ${PHYDLL_BUILD_DIR}/include)
    target_link_libraries(main_cnn_test_3D.x PRIVATE ${PHYDLL_BUILD_DIR}/lib/libphydll_f.so)
    target_link_libraries(main_cnn_test_3D.x PRIVATE ${PHYDLL_BUILD_DIR}/lib/libphydll.so)
endif()

install(
    TARGETS main_cnn_test_3D.x
    RUNTIME DESTINATION ${CMAKE_BINARY_DIR}/bin
)

if(WITH_SCOREP)
    scorep_instrument_target(
        main_cnn_test_3D.x
            USER ON
            COMPILER ON
    )
endif()

# Test 3D Convoultion Mulichannel Executable
add_executable(main_cnn_test_3D_multichannel.x main_cnn_test_3D_multichannel.f90)

target_sources(
    main_cnn_test_3D_multichannel.x
    PRIVATE
    main_cnn_test_3D_multichannel.f90
        ${CMAKE_SOURCE_DIR}/src/ml_coupling/ml_coupling.F90
        ${CMAKE_SOURCE_DIR}/src/ml_coupling/ml_coupling_combustion.F90
        ${CMAKE_SOURCE_DIR}/src/ml_coupling_strategy/ml_coupling_strategy.f90
)

target_include_directories(main_cnn_test_3D_multichannel.x PRIVATE ${MPI_Fortran_INCLUDE_DIRS})
target_link_libraries(main_cnn_test_3D_multichannel.x PRIVATE ${MPI_LIBRARIES})

if(WITH_AIX)
    target_sources(
        main_cnn_test_3D_multichannel.x
        PRIVATE
            ${CMAKE_SOURCE_DIR}/src/ml_coupling_strategy/aix/ml_coupling_strategy_aix.f90
    )
    # aixelerator_service_mod comes from here (maybe find a better place in the AIxeleratorService project for it)
    target_include_directories(main_cnn_test_3D_multichannel.x PRIVATE ${AIX_BUILD_DIR}/test)
    target_link_libraries(main_cnn_test_3D_multichannel.x PRIVATE ${AIX_BUILD_DIR}/lib/libAIxeleratorService.so)
    target_link_libraries(main_cnn_test_3D_multichannel.x PRIVATE ${TENSORFLOW_LIBRARIES})
endif()
if(WITH_PHYDLL)
    target_sources(
        main_cnn_test_3D_multichannel.x
        PRIVATE
            ${CMAKE_SOURCE_DIR}/src/ml_coupling_strategy/phydll/ml_coupling_strategy_phydll.F90
    )
    target_include_directories(main_cnn_test_3D_multichannel.x PRIVATE ${PHYDLL_BUILD_DIR}/include)
    target_link_libraries(main_cnn_test_3D_multichannel.x PRIVATE ${PHYDLL_BUILD_DIR}/lib/libphydll_f.so)
    target_link_libraries(main_cnn_test_3D_multichannel.x PRIVATE ${PHYDLL_BUILD_DIR}/lib/libphydll.so)
endif()

install(
    TARGETS main_cnn_test_3D_multichannel.x
    RUNTIME DESTINATION ${CMAKE_BINARY_DIR}/bin
)

if(WITH_SCOREP)
    scorep_instrument_target(
        main_cnn_test_3D_multichannel.x
            USER ON
            COMPILER ON
    )
endif()

# Test TSRGAN 3D
add_executable(main_tsrgan.x main_tsrgan.f90)

target_sources(
    main_tsrgan.x
    PRIVATE
        main_tsrgan.f90
        ${CMAKE_SOURCE_DIR}/src/ml_coupling/ml_coupling.F90
        ${CMAKE_SOURCE_DIR}/src/ml_coupling/ml_coupling_turbulence.F90
        ${CMAKE_SOURCE_DIR}/src/ml_coupling_strategy/ml_coupling_strategy.f90
)

target_include_directories(
    main_tsrgan.x 
    PRIVATE 
        ${MPI_Fortran_INCLUDE_DIRS}
        ${CMAKE_SOURCE_DIR}/extern/h5fortran/BUILD/INSTALL/include
)

target_link_libraries(main_tsrgan.x PRIVATE h5fortran::h5fortran)
target_link_libraries(main_tsrgan.x PRIVATE ${MPI_LIBRARIES})


if(WITH_AIX)
    target_sources(
        main_tsrgan.x
        PRIVATE
            ${CMAKE_SOURCE_DIR}/src/ml_coupling_strategy/aix/ml_coupling_strategy_aix.f90
    )
    # aixelerator_service_mod comes from here (maybe find a better place in the AIxeleratorService project for it)
    target_include_directories(main_tsrgan.x PRIVATE ${AIX_BUILD_DIR}/test)
    target_link_libraries(main_tsrgan.x PRIVATE ${AIX_BUILD_DIR}/lib/libAIxeleratorService.so)
    target_link_libraries(main_tsrgan.x PRIVATE ${TENSORFLOW_LIBRARIES})
endif()

if(WITH_PHYDLL)
    target_sources(
        main_tsrgan.x
        PRIVATE
            ${CMAKE_SOURCE_DIR}/src/ml_coupling_strategy/phydll/ml_coupling_strategy_phydll.F90
    )
    target_include_directories(main_tsrgan.x PRIVATE ${PHYDLL_BUILD_DIR}/include)
    target_link_libraries(main_tsrgan.x PRIVATE ${PHYDLL_BUILD_DIR}/lib/libphydll_f.so)
    target_link_libraries(main_tsrgan.x PRIVATE ${PHYDLL_BUILD_DIR}/lib/libphydll.so)
endif()

install(
    TARGETS  main_tsrgan.x
    RUNTIME DESTINATION ${CMAKE_BINARY_DIR}/bin
)
