#pragma once

namespace bt::ops {

struct Add {
  float operator()(float x, float y) const noexcept { return x + y; }
};
struct Sub {
  float operator()(float x, float y) const noexcept { return x - y; }
};
struct Mul {
  float operator()(float x, float y) const noexcept { return x * y; }
};
struct Div {
  float operator()(float x, float y) const noexcept { return x / y; }
};

}  // namespace bt::ops
