# ======================================================================== #
# Copyright 2018-2019 Ingo Wald                                            #
#                                                                          #
# Licensed under the Apache License, Version 2.0 (the "License");          #
# you may not use this file except in compliance with the License.         #
# You may obtain a copy of the License at                                  #
#                                                                          #
#     http://www.apache.org/licenses/LICENSE-2.0                           #
#                                                                          #
# Unless required by applicable law or agreed to in writing, software      #
# distributed under the License is distributed on an "AS IS" BASIS,        #
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. #
# See the License for the specific language governing permissions and      #
# limitations under the License.                                           #
# ======================================================================== #

find_package(OpenGL REQUIRED)

include_directories(${OptiX_INCLUDE})

cuda_compile_and_embed(embedded_visualizer_code CudaSonelVisualizer.cu)
CUDA_COMPILE_AND_EMBED(embedded_mapper_code CudaSonelMapper.cu)

add_executable(sonel-mapping
  ${embedded_visualizer_code}
  ${embedded_mapper_code}
  
  Camera.h
  CameraFrame.h
  CameraFrame.cpp
  CameraFrameManip.h
  CameraFrameManip.cpp
  FlyModeManip.h
  FlyModeManip.cpp
  InspectModeManip.h
  InspectModeManip.cpp
  
  GlfCameraWindow.h
  GlfCameraWindow.cpp
  GlfWindow.h
  GlfWindow.cpp
  MainWindow.h
  MainWindow.cpp

  LaunchParams.h
  Model.h
  Model.cpp
  TriangleMeshSbtData.h

  CudaBuffer.h
  CudaHelper.h
  CudaRandom.h
  CudaSonelMapperParams.h
  
  OptixScene.h
  OptixScene.cpp
  OptixSetup.h
  OptixSetup.cpp
 
  Sonel.h
  SoundSource.h
  SonelMap.h
  OctTree.h

  SonelManager.h
  SonelManager.cpp
  SonelMapper.h
  SonelMapper.cpp
  SonelMapVisualizer.h
  SonelMapVisualizer.cpp

  main.cpp
  
  )

target_link_libraries(
sonel-mapping
  gdt
  # optix dependencies, for rendering
  ${optix_LIBRARY}
  ${CUDA_LIBRARIES}
  ${CUDA_CUDA_LIBRARY}
  # glfw and opengl, for display
  glfw
  ${OPENGL_gl_LIBRARY}
  )
