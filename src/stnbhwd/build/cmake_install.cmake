# Install script for directory: /Users/chenbaian/Desktop/project/SSFlow/ssflow_release/src/stnbhwd

# Set the install prefix
if(NOT DEFINED CMAKE_INSTALL_PREFIX)
  set(CMAKE_INSTALL_PREFIX "/Users/chenbaian/torch/install")
endif()
string(REGEX REPLACE "/$" "" CMAKE_INSTALL_PREFIX "${CMAKE_INSTALL_PREFIX}")

# Set the install configuration name.
if(NOT DEFINED CMAKE_INSTALL_CONFIG_NAME)
  if(BUILD_TYPE)
    string(REGEX REPLACE "^[^A-Za-z0-9_]+" ""
           CMAKE_INSTALL_CONFIG_NAME "${BUILD_TYPE}")
  else()
    set(CMAKE_INSTALL_CONFIG_NAME "Release")
  endif()
  message(STATUS "Install configuration: \"${CMAKE_INSTALL_CONFIG_NAME}\"")
endif()

# Set the component getting installed.
if(NOT CMAKE_INSTALL_COMPONENT)
  if(COMPONENT)
    message(STATUS "Install component: \"${COMPONENT}\"")
    set(CMAKE_INSTALL_COMPONENT "${COMPONENT}")
  else()
    set(CMAKE_INSTALL_COMPONENT)
  endif()
endif()

if("${CMAKE_INSTALL_COMPONENT}" STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/lib/luarocks/rocks/stnbhwd/scm-1/lib" TYPE MODULE FILES "/Users/chenbaian/Desktop/project/SSFlow/ssflow_release/src/stnbhwd/build/libstn.so")
  if(EXISTS "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/lib/luarocks/rocks/stnbhwd/scm-1/lib/libstn.so" AND
     NOT IS_SYMLINK "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/lib/luarocks/rocks/stnbhwd/scm-1/lib/libstn.so")
    execute_process(COMMAND /usr/bin/install_name_tool
      -delete_rpath "/Users/chenbaian/torch/install/lib"
      "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/lib/luarocks/rocks/stnbhwd/scm-1/lib/libstn.so")
    execute_process(COMMAND /usr/bin/install_name_tool
      -add_rpath "@executable_path/../lib"
      "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/lib/luarocks/rocks/stnbhwd/scm-1/lib/libstn.so")
    execute_process(COMMAND /usr/bin/install_name_tool
      -add_rpath "/Users/chenbaian/torch/install/lib"
      "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/lib/luarocks/rocks/stnbhwd/scm-1/lib/libstn.so")
    if(CMAKE_INSTALL_DO_STRIP)
      execute_process(COMMAND "/Applications/Xcode.app/Contents/Developer/Toolchains/XcodeDefault.xctoolchain/usr/bin/strip" "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/lib/luarocks/rocks/stnbhwd/scm-1/lib/libstn.so")
    endif()
  endif()
endif()

if("${CMAKE_INSTALL_COMPONENT}" STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/lib/luarocks/rocks/stnbhwd/scm-1/lua/stn" TYPE FILE FILES
    "/Users/chenbaian/Desktop/project/SSFlow/ssflow_release/src/stnbhwd/AffineGridGeneratorBHWD.lua"
    "/Users/chenbaian/Desktop/project/SSFlow/ssflow_release/src/stnbhwd/AffineTransformMatrixGenerator.lua"
    "/Users/chenbaian/Desktop/project/SSFlow/ssflow_release/src/stnbhwd/BilinearSamplerBHWD.lua"
    "/Users/chenbaian/Desktop/project/SSFlow/ssflow_release/src/stnbhwd/ScaleBHWD.lua"
    "/Users/chenbaian/Desktop/project/SSFlow/ssflow_release/src/stnbhwd/init.lua"
    "/Users/chenbaian/Desktop/project/SSFlow/ssflow_release/src/stnbhwd/test.lua"
    )
endif()

if(CMAKE_INSTALL_COMPONENT)
  set(CMAKE_INSTALL_MANIFEST "install_manifest_${CMAKE_INSTALL_COMPONENT}.txt")
else()
  set(CMAKE_INSTALL_MANIFEST "install_manifest.txt")
endif()

string(REPLACE ";" "\n" CMAKE_INSTALL_MANIFEST_CONTENT
       "${CMAKE_INSTALL_MANIFEST_FILES}")
file(WRITE "/Users/chenbaian/Desktop/project/SSFlow/ssflow_release/src/stnbhwd/build/${CMAKE_INSTALL_MANIFEST}"
     "${CMAKE_INSTALL_MANIFEST_CONTENT}")
