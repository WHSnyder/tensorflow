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
#include "tensorflow/lite/delegates/gpu/gl_delegate.h"

namespace tflite {

namespace {

int num_delegates_created = 0;
int num_delegates_destroyed = 0;
int num_delegates_invoked = 0;
int options_counter = 0;
int (*destruction_callback)(const char* s) = nullptr;
typedef void (*ErrorHandler)(const char*);

}  // namespace

extern "C" {
TfLiteDelegate* tflite_plugin_create_delegate(char** options_keys,
                                              char** options_values,
                                              size_t num_options,
                                              ErrorHandler error_handler) {
  //auto opts = TfLiteGpuDelegateOptionsDefault();

  TfLiteDelegate* ptr = TfLiteGpuDelegateCreate(nullptr);

  return ptr;
}

void set_destroy_callback(int (*callback)(const char* s)) {
  destruction_callback = callback;
}

void tflite_plugin_destroy_delegate(TfLiteDelegate* delegate) {
  TfLiteGpuDelegateDelete(delegate);
}

void initialize_counters() {
  num_delegates_created = 0;
  num_delegates_destroyed = 0;
  num_delegates_invoked = 0;
  options_counter = 0;
}

int get_num_delegates_created() { return num_delegates_created; }

int get_num_delegates_destroyed() { return num_delegates_destroyed; }

int get_num_delegates_invoked() { return num_delegates_invoked; }

int get_options_counter() { return options_counter; }
}

}  // namespace tflite
