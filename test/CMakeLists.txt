LIST(APPEND CMAKE_MODULE_PATH "${CMAKE_SOURCE_DIR}/catboost") 
set(flatbuffers_home "${CMAKE_SOURCE_DIR}/../flatbuffers")
set(catboost_home "${CMAKE_SOURCE_DIR}/../catboost")

find_package(Boost COMPONENTS system filesystem unit_test_framework REQUIRED)
find_package(flatbuffers REQUIRED)
find_package(catboost REQUIRED)

include_directories(${flatbuffers_INCLUDE_DIRS})
include_directories(${catboost_INCLUDE_DIRS})
include_directories (${cu_hlt_SOURCE_DIR}/catboost/include) 

ADD_DEFINITIONS(-DBOOST_TEST_DYN_LINK) 

find_package(CUDA REQUIRED)
#set(CUDA_HOST_COMPILER "g++")
set(CUDA_SEPARABLE_COMPILATION ON)
option(CUDA_PROPAGATE_HOST_FLAGS OFF)

# Cuda: Deal with build type
if(${CMAKE_BUILD_TYPE} STREQUAL RelWithDebInfo)
  set(CUDA_BUILD_TYPE "-O3\;-g\;-DNDEBUG\;")
elseif(${CMAKE_BUILD_TYPE} STREQUAL Release)
  set(CUDA_BUILD_TYPE "-O3\;-DNDEBUG\;")
elseif(${CMAKE_BUILD_TYPE} STREQUAL Debug)
  set(CUDA_BUILD_TYPE "-O0\;-G\;-g\;-DDEBUG\;")
endif(${CMAKE_BUILD_TYPE} STREQUAL RelWithDebInfo)

set(CUDA_NVCC_FLAGS "${CUDA_BUILD_TYPE}--use_fast_math;--expt-relaxed-constexpr;--generate-line-info;--resource-usage;--verbose;--nvlink-options=--verbose;-Xptxas=--verbose;--maxrregcount=63;")

enable_testing()
file(GLOB TEST_SRCS "${CMAKE_CURRENT_SOURCE_DIR}/*.cpp")
file(GLOB TEST_SRCS_CU "${CMAKE_CURRENT_SOURCE_DIR}/*.cu")

#Run through each source
foreach(testSrc ${TEST_SRCS})
        #Extract the filename without an extension (NAME_WE)
        get_filename_component(testName ${testSrc} NAME_WE)

        #Add compile target
        add_executable(${testName} ${testSrc})

        #link to Boost libraries AND your targets and dependencies
        target_link_libraries(${testName} ${Boost_LIBRARIES} Catboost)

        #I like to move testing binaries into a testBin directory
        set_target_properties(${testName} PROPERTIES 
            RUNTIME_OUTPUT_DIRECTORY  ${CMAKE_CURRENT_SOURCE_DIR}/testBin)

        #Finally add it to test execution - 
        #Notice the WORKING_DIRECTORY and COMMAND
        add_test(NAME ${testName} 
                 WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/testBin 
                 COMMAND ${CMAKE_CURRENT_SOURCE_DIR}/testBin/${testName} )
endforeach(testSrc)

foreach(testSrc_cu ${TEST_SRCS_CU})
        #Extract the filename without an extension (NAME_WE)
        get_filename_component(testName ${testSrc_cu} NAME_WE)

        #Add compile target
        cuda_add_executable(${testName} ${testSrc_cu})

        #link to Boost libraries AND your targets and dependencies
        target_link_libraries(${testName} ${Boost_LIBRARIES} Catboost Evaluator)

        #I like to move testing binaries into a testBin directory
        set_target_properties(${testName} PROPERTIES 
            RUNTIME_OUTPUT_DIRECTORY  ${CMAKE_CURRENT_SOURCE_DIR}/testBin)

        #Finally add it to test execution - 
        #Notice the WORKING_DIRECTORY and COMMAND
        add_test(NAME ${testName} 
                 WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/testBin 
                 COMMAND ${CMAKE_CURRENT_SOURCE_DIR}/testBin/${testName} )
endforeach(testSrc_cu)