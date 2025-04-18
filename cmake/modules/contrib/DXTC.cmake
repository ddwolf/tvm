# Licensed to the Apache Software Foundation (ASF) under one
# or more contributor license agreements.  See the NOTICE file
# distributed with this work for additional information
# regarding copyright ownership.  The ASF licenses this file
# to you under the Apache License, Version 2.0 (the
# "License"); you may not use this file except in compliance
# with the License.  You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing,
# software distributed under the License is distributed on an
# "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
# KIND, either express or implied.  See the License for the
# specific language governing permissions and limitations
# under the License.

if((USE_DXTC STREQUAL "ON"))
  tvm_file_glob(GLOB DNNL_CONTRIB_SRC src/relay/backend/contrib/dxtc/*.cc)
  list(APPEND COMPILER_SRCS ${DNNL_CONTRIB_SRC})

  message(STATUS "Build with DXTC compiler")
else()
  message(FATAL_ERROR "Invalid option: USE_DXTC=" ${USE_DXTC})
endif()
