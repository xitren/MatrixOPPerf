cmake_minimum_required(VERSION 3.16)

project(CUDACMAKETEST)

# set(CMAKE_MODULE_PATH "/usr/share/cmake/Modules" ${CMAKE_MODULE_PATH})

find_package(CUDA)

if (CUDA_FOUND)
	message("CUDA found, using device squaring!")

        # Append current NVCC flags by something, eg comput capability
        set(CUDA_NVCC_FLAGS ${CUDA_NVCC_FLAGS} --gpu-architecture sm_75)

        set(CUDASRCS 
                main.cu 
        )

        CUDA_ADD_EXECUTABLE(gpuSquareDemo ${CUDASRCS})

else()
	message("CUDA not found, doing something alternatively")
endif()