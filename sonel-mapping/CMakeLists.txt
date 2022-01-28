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

cuda_compile_and_embed(embedded_visualizer_code Cuda/CudaSonelVisualizer.cu)
CUDA_COMPILE_AND_EMBED(embedded_mapper_code Cuda/CudaSonelMapper.cu)
CUDA_COMPILE_AND_EMBED(embedded_receiver_code Cuda/CudaSonelReceiver.cu)

add_executable(sonel-mapping
  ${embedded_visualizer_code}
  ${embedded_mapper_code}
        ${embedded_receiver_code}

        UI/Camera.h
        UI/CameraFrame.h
        UI/CameraFrame.cpp
        UI/CameraFrameManip.h
        UI/CameraFrameManip.cpp
        UI/FlyModeManip.h
        UI/FlyModeManip.cpp
        UI/InspectModeManip.h
        UI/InspectModeManip.cpp

        UI/GlfCameraWindow.h
        UI/GlfCameraWindow.cpp
        UI/GlfWindow.h
        UI/GlfWindow.cpp
        UI/MainWindow.h
        UI/MainWindow.cpp

        SonelMapping/Models/SonelVisualizerParams.h
        SonelMapping/Models/Model.h
        SonelMapping/Models/Model.cpp
        SonelMapping/Models/TriangleMeshSbtData.h

        Cuda/CudaBuffer.h
        Cuda/CudaHelper.h
        Cuda/CudaRandom.h
        Cuda/CudaSonelMapperParams.h

        SonelMapping/OptixScene.h
        SonelMapping/OptixScene.cpp
        SonelMapping/OptixSetup.h
        SonelMapping/OptixSetup.cpp

        SonelMapping/Models/Sonel.h
        SonelMapping/Models/SoundSource.h
        SonelMapping/Models/SonelMap.h

        SonelMapping/SonelManager.h
        SonelMapping/SonelManager.cpp
        SonelMapping/SonelMapVisualizer.h
        SonelMapping/SonelMapVisualizer.cpp

  main.cpp

        Cuda/CudaVector.h SonelMapping/SonelVisibilityFlags.h SonelMapping/SmOptixProgram.h SonelMapping/SonelMapper.cpp SonelMapping/SonelMapper.h SonelMapping/Models/SonelReceiverParams.h SonelMapping/SonelMapReceiver.cpp SonelMapping/SonelMapReceiver.h)

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
