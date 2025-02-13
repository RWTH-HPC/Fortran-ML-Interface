add_library( mlCouplingLib SHARED)
set_target_properties(mlCouplingLib PROPERTIES OUTPUT_NAME mlCoupling)
if(WITH_AIX)
    target_sources(
        mlCouplingLib 
            PRIVATE 
                )
endif()
if(WITH_PHYDLL)
    target_sources(
        mlCouplingLib 
            PRIVATE 
                )
endif()
target_sources(
    mlCouplingLib
        PUBLIC
            ml_coupling_strategy/phydll/ml_coupling_strategy_phydll.F90
            ml_coupling_strategy/aix/ml_coupling_strategy_aix.f90
            ml_coupling_strategy/ml_coupling_strategy.f90
            ml_coupling/ml_coupling.F90
            ml_coupling/ml_coupling_combustion.F90
            ml_coupling/ml_coupling_turbulence.F90
)

target_include_directories(
    mlCouplingLib
        PUBLIC  
            ${AIX_BUILD_DIR}/test
            ${PHYDLL_BUILD_DIR}/include
            ${CMAKE_SOURCE_DIR}/extern/h5fortran/BUILD/INSTALL/include
)

#target_link_libraries(mlCouplingLib PRIVATE h5fortran::h5fortran)
target_link_libraries(
    mlCouplingLib
        PRIVATE 
        ${AIX_BUILD_DIR}/lib/libAIxeleratorService.so   
        ${PHYDLL_BUILD_DIR}/lib/libphydll_f.so   
        ${PHYDLL_BUILD_DIR}/lib/libphydll.so  
)

if(WITH_SCOREP)
    scorep_instrument_target(
        mlCouplingLib
            USER ON
            COMPILER ON
    )
endif()

install(
    TARGETS mlCouplingLib
    LIBRARY DESTINATION ${CMAKE_BINARY_DIR}/lib
    #PUBLIC_HEADER DESTINATION ${CMAKE_BINARY_DIR}/include
)