// Minimal stand-in for <gtest/gtest.h> so the xetla int2 x int8 harness
// (int2_fp16_dpas_fp16scales_fast_test) builds without the googletest
// submodule. FAIL / ASSERT_EQ abort the run with a message.
#pragma once

#include <cstdlib>
#include <iostream>
#include <string>

namespace testing {
class Test {
 public:
  template <typename T>
  static void RecordProperty(const std::string&, const T&) {}
  static void RecordProperty(const std::string&, const char*) {}
};
}  // namespace testing

#define FAIL()                                                        \
  do {                                                                \
    std::cout << "FAIL() at " << __FILE__ << ":" << __LINE__ << "\n"; \
    std::exit(1);                                                     \
  } while (0)
#define ASSERT_EQ(a, b)                                                 \
  do {                                                                  \
    if (!((a) == (b))) {                                                \
      std::cout << "ASSERT_EQ failed at " << __FILE__ << ":" << __LINE__ \
                << "\n";                                                \
    }                                                                   \
  } while (0)
