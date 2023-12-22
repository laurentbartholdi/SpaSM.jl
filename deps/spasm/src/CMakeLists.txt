#add_compile_options(-Werror)

add_library(spasm SHARED
	# crypto stuff
	spasm_prng.c sha256.c

	# arithmetic
	spasm_ZZp.c
	
	# common stuff, IO, utilities
	spasm_util.c spasm_triplet.c spasm_io.c
	spasm_scatter.c spasm_spmv.c 
	spasm_transpose.c spasm_permutation.c

	# triangular solver
	spasm_reach.c spasm_triangular.c 
	
	# echelonization
	spasm_pivots.c spasm_schur.c spasm_ffpack.cpp 
	spasm_echelonize.c 

	# main functionnalities
	spasm_solve.c spasm_kernel.c spasm_rref.c spasm_certificate.c
	
	# dulmage-mendelson
	spasm_submatrix.c spasm_matching.c spasm_dm.c spasm_scc.c 
)

target_link_libraries(spasm PUBLIC OpenMP::OpenMP_C)
target_link_libraries(spasm PUBLIC m)
target_include_directories(spasm PUBLIC $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/../../usr/include>)
target_link_directories(spasm PUBLIC $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/../../usr/lib>)
target_link_libraries(spasm PUBLIC PkgConfig::GMP)
target_link_libraries(spasm PUBLIC givaro blas)
#target_link_libraries(spasm PUBLIC PkgConfig::GIVARO)
#target_link_libraries(spasm PUBLIC PkgConfig::FFLAS_FFPACK)
