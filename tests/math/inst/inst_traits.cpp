// tests/math/inst/inst_traits.cpp — verifies the promotion rules in isolation.
#include <vector> // pre-existing TypeUtil.hpp dependency (Container<T> uses std::vector) not self-included
#define MATSAR_UTILS // bypass UtilHeadCheck.hpp's umbrella-only guard for this isolated header test
#include "MatSAR/util/TypeUtil.hpp"
#include <complex>
#include <type_traits>
using namespace matsar;

static_assert(std::is_same_v<resolve_t<void, float64>, float64>, "resolve void->fallback");
static_assert(std::is_same_v<resolve_t<float32, float64>, float32>, "resolve override wins");

static_assert(std::is_same_v<accumulate_t<int8>,  int64>,  "accum int8->int64");
static_assert(std::is_same_v<accumulate_t<int32>, int64>,  "accum int32->int64");
static_assert(std::is_same_v<accumulate_t<uint16>, uint64>,"accum uint16->uint64");
static_assert(std::is_same_v<accumulate_t<float32>, float32>, "accum f32 keep");
static_assert(std::is_same_v<accumulate_t<complex64>, complex64>, "accum c64 keep");

static_assert(std::is_same_v<real_promote_t<int32>, real_max>, "realpromote int->real_max");
static_assert(std::is_same_v<real_promote_t<float32>, float32>, "realpromote f32 keep");
static_assert(std::is_same_v<real_promote_t<complex64>, complex64>, "realpromote c64 keep");

static_assert(std::is_same_v<magnitude_t<int32>, real_max>, "mag int->real_max");
static_assert(std::is_same_v<magnitude_t<float32>, float32>, "mag f32 keep");
static_assert(std::is_same_v<magnitude_t<complex64>, float32>, "mag c64->f32");
static_assert(std::is_same_v<magnitude_t<complex128>, float64>, "mag c128->f64");

static_assert(std::is_same_v<linalg_promote_t<int32>, real_max>, "la int->real_max");
static_assert(std::is_same_v<linalg_promote_t<float32>, real_max>, "la f32->real_max");
static_assert(std::is_same_v<linalg_promote_t<complex64>, complex_max>, "la c64->complex_max");

int main() { return 0; }
