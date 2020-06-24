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

#include "tensorflow/lite/delegates/gpu/gl/kernels/mean_reduce.h"

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

class MeanReduce : public NodeShader {
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

    std::string source = R"(
      //MeanReduce


      //workgroup test size = 16
      int inputSize = int($input_data_0_w$ * $input_data_0_h$);
      shared float tempSums[floor($input_data_0_w$ * $input_data_0_h$ / 16) + 1];
      
      
      

      /*
      highp vec4 sum = vec4(0.0);
      highp float size = float($input_data_0_w$ * $input_data_0_h$);
      for (int w = 0; w < $input_data_0_w$; w++) {
        for (int h = 0; h < $input_data_0_h$; h++) {
          sum += $input_data_0[w, h, gid.z]$;
        }
      }

      value_0 = sum / size;
      */
    )";
    *generated_code = {
        /*parameters=*/std::move(parameters),
        /*objects=*/{},
        /*shared_variables=*/{},
        /*workload=*/uint3(),
        /*workgroup=*/uint3(16, 16, 4),
        /*source_code=*/std::move(source),
        /*input=*/IOStructure::ONLY_DEFINITIONS,
        /*output=*/IOStructure::ONLY_DEFINITIONS,
    };
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
