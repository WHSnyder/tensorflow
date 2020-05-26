/* Copyright 2019 The TensorFlow Authors. All Rights Reserved.

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
#include <cstdio>
#include <cstdlib>
#include <cstring>

#include "tensorflow/lite/c/common.h"
#include "tensorflow/lite/delegates/gpu/metal_delegate.h"

namespace tflite {

typedef void (*ErrorHandler)(const char*);


extern "C" {
TFL_CAPI_EXPORT TfLiteDelegate* tflite_plugin_create_delegate(char** options_keys,
                                              char** options_values,
                                              size_t num_options,
                                              ErrorHandler error_handler);

TFL_CAPI_EXPORT void tflite_plugin_destroy_delegate(TfLiteDelegate* delegate);
}


}
