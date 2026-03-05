cmake_minimum_required(VERSION 3.19)
project(mobile_lm_store)

set(CMAKE_CXX_STANDARD 17)

message(STATUS "Check sqlite...")
execute_process(
    COMMAND bash ${CMAKE_CURRENT_LIST_DIR}/vendor.sh
    WORKING_DIRECTORY ${CMAKE_CURRENT_LIST_DIR}
)

include_directories(
                    # sqlite include
                    ${CMAKE_CURRENT_LIST_DIR}/vendor/
                    ${CMAKE_CURRENT_LIST_DIR}/include/
                    )

FILE(GLOB LMStore_Srcs "${CMAKE_CURRENT_LIST_DIR}/src/*.cpp")

add_library(LMStore SHARED
            ${CMAKE_CURRENT_LIST_DIR}/vendor/sqlite3.c
            ${LMStore_Srcs})
add_definitions(-DSQLITE_CORE)