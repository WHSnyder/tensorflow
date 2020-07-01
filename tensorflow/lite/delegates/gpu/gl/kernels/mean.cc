/* Copyright 2020 The TensorFlow Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
==============================================================================*/

#include "tensorflow/lite/delegates/gpu/gl/kernels/mean.h"

#include <algorithm>
#include <cstdint>
#include <cstring>
#include <string>
#include <vector>

#include "absl/memory/memory.h"
#include "tensorflow/lite/delegates/gpu/common/status.h"
#include "tensorflow/lite/delegates/gpu/common/types.h"

namespace tflite {
namespace gpu {
namespace gl {
namespace {

class Mean : public NodeShader {
 public:
  Status GenerateCode(const GenerationContext& ctx,
                      GeneratedCode* generated_code) const final {
    auto attr = absl::any_cast<MeanAttributes>(ctx.node->operation.attributes);
    if (attr.dims != std::set<Axis>({Axis::HEIGHT, Axis::WIDTH})) {
      return InvalidArgumentError(
          "Mean calculation is supported only for height and width.");
    }

    auto input = ctx.graph->FindInputs(ctx.node->id)[0];

    std::vector<Variable> parameters = {
        {"input_data_0_h", input->tensor.shape.h},
        {"input_data_0_w", input->tensor.shape.w},
        {"input_data_0_c", input->tensor.shape.c}};


    std::string source;

    constexpr int kWorkgroupHintX = 4;
    constexpr int kWorkgroupHintY = 4;
    int kWorkgroupHintZ = ceil(input->tensor.shape.c / 4.0 - .0001);

    std::vector<Variable> shared_variables = {
        {"sh_mem", std::vector<float4>(kWorkgroupHintX/2 * kWorkgroupHintX * 
                                       kWorkgroupHintY/2 * kWorkgroupHintY * 
                                       kWorkgroupHintZ)},
    };


    if (input->tensor.shape.h * input->tensor.shape.w >= 1024) { //Arbitrary

      source = R"(        
        /*MEAN*/
        highp vec4 sum = vec4(0.0);
        highp float size = float($input_data_0_w$ * $input_data_0_h$);

        const int groupsX = int(gl_NumWorkGroups.x);
        const int groupsY = int(gl_NumWorkGroups.y);
        const int sizeX = int(gl_WorkGroupSize.x);
        const int sizeY = int(gl_WorkGroupSize.y);
        ivec3 localID = ivec3(gl_LocalInvocationID.xyz);
        int workGridSize = sizeX * sizeY * groupsX * groupsY;

        int local_grid_index = sizeX * localID.y + localID.x;
        int global_grid_index = gid.x * sizeX * sizeY * groupsY + gid.y * sizeX * sizeY;
        
        int taskSize = int(ceil(size / float(workGridSize)));

        int flattenedIndex = global_grid_index + local_grid_index;
        int z_offset = gid.z * int(size);

        int startIndex = flattenedIndex * taskSize;

        for (int i = startIndex; i < startIndex + taskSize; i++) {
          sum += i < size ? $input_data_0[z_offset + i, 0, 0]$ : vec4(0.0);
        }

        z_offset = gid.z * workGridSize;

        sh_mem[flattenedIndex + z_offset] = sum;

        memoryBarrierShared();
        groupMemoryBarrier();
        barrier();

        if (gid.x >= 1 || gid.y >= 1){
          return;
        }

        sum = vec4(0.0);
        
        for (int i = 0; i < workGridSize; i++){
          sum += sh_mem[i + z_offset];  
        }

        value_0 = sum / size;
      )";

      *generated_code = {
          /*parameters=*/std::move(parameters),
          /*objects=*/{},
          /*shared_variables=*/std::move(shared_variables),
          /*workload=*/uint3(kWorkgroupHintX,kWorkgroupHintY,1),
          /*workgroup=*/ uint3(kWorkgroupHintX/2,kWorkgroupHintY/2,kWorkgroupHintZ), 
          /*source_code=*/std::move(source),
          /*input=*/IOStructure::ONLY_DEFINITIONS,
          /*output=*/IOStructure::AUTO,
      };
    }

    else {

      source = R"(        
<<<<<<< Updated upstream
=======
        
        /*Shaders may be compiled with a precision hint mediump, which means that
        GLSL compiler may drop the size of float data type from 32 to 16 bits.
        If "sum" and "size" variables are 16bit floats, their values range
        become not enough for providing a good results accuracy. That is why
        their precision is forced to be 32bit by using highp qualifier.*/
        
>>>>>>> Stashed changes
        highp vec4 sum = vec4(0.0);
        highp float size = float($input_data_0_w$ * $input_data_0_h$);
        for (int h = 0; h < $input_data_0_h$; h++) {
          for (int w = 0; w < $input_data_0_w$; w++) {
            sum += $input_data_0[w, h, gid.z]$;
          }
        }

        value_0 = sum / size;
      )";
    
      *generated_code = {
          /*parameters=*/std::move(parameters),
          /*objects=*/{},
          /*shared_variables=*/{},
          /*workload=*/uint3(),
          /*workgroup=*/uint3(),
          /*source_code=*/std::move(source),
          /*input=*/IOStructure::ONLY_DEFINITIONS,
          /*output=*/IOStructure::AUTO,
      };
    }
    return OkStatus();
  }
};

}  // namespace

std::unique_ptr<NodeShader> NewMeanNodeShader() {
  return absl::make_unique<Mean>();
}

}  // namespace gl
}  // namespace gpu
}  // namespace tflite
