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
        {"input_data_0_w", input->tensor.shape.w}};

    /*
    Shaders may be compiled with a precision hint mediump, which means that
    GLSL compiler may drop the size of float data type from 32 to 16 bits.
    If "sum" and "size" variables are 16bit floats, their values range
    become not enough for providing a good results accuracy. That is why
    their precision is forced to be 32bit by using highp qualifier.
    */

    std::string source;

    if (input->tensor.shape.h * input->tensor.shape.w <= 256){

      source = R"(        
        /*MEAN*/
        highp vec4 sum = vec4(0.0);
        highp float size = float($input_data_0_w$ * $input_data_0_h$);

        const int threads = int(gl_WorkGroupSize.y);
        const int workers = int(gl_WorkGroupSize.x);
        ivec3 tid = ivec3(gl_LocalInvocationID);
   
        int localSize = int(ceil(size/256.0));
        
        int x = tid.x;
        int y = tid.y;

        int start = x * (threads * localSize) + y * localSize;

        int z_offset = tid.z * size;

        for (int i = 0; i < localSize; i++) {
          int index = start + i + z_offset;
          sum += start < size ? $input_data_0[index, 0, 0]$ : vec4(0.0);
        }

        value_0 = (sum / size);
        sh_mem[tid.x * tid.y + tid.x] = value_0

        memoryBarrierShared();
        barrier();

        if (gid.x >= 1 || gid.y >= 1 || gid.z >= 4){
          return;
        }

        for (int i = 0; i < 16; i++){
          
        }
      )";

      *generated_code = {
          /*parameters=*/std::move(parameters),
          /*objects=*/{},
          /*shared_variables=*/{"sh_mem", std::vector<float4>(0)},
          /*workload=*/uint3(1,1,1),
          /*workgroup=*/uint3(16,16,4),
          /*source_code=*/std::move(source),
          /*input=*/IOStructure::ONLY_DEFINITIONS,
          /*output=*/IOStructure::AUTO,
      };
    } else {

      source = R"(        
        /*MEAN*/
        highp vec4 sum = vec4(0.0);
        highp float size = float($input_data_0_w$ * $input_data_0_h$);
        for (int h = 0; h < $input_data_0_h$; h+=2) {
          for (int w = 0; w < $input_data_0_w$; w+=2) {
            sum += $input_data_0[w, h, gid.z]$;
          }
        }

        value_0 = (sum / size) * 4.0;
      )";
    
      *generated_code = {
          /*parameters=*/std::move(parameters),
          /*objects=*/{},
          /*shared_variables=*/{/*"sh_mem", std::vector<float4>(0)*/},
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
