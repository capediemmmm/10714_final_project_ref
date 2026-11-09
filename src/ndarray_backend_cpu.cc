#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <stdexcept>
#include <utility>
#include <vector>

namespace needle {
namespace cpu {

#define ALIGNMENT 256
#define TILE 8
typedef float scalar_t;
const size_t ELEM_SIZE = sizeof(scalar_t);


/**
 * This is a utility structure for maintaining an array aligned to ALIGNMENT boundaries in
 * memory.  This alignment should be at least TILE * ELEM_SIZE, though we make it even larger
 * here by default.
 */
struct AlignedArray {
  AlignedArray(const size_t size) {
    int ret = posix_memalign((void**)&ptr, ALIGNMENT, size * ELEM_SIZE);
    if (ret != 0) throw std::bad_alloc();
    this->size = size;
  }
  ~AlignedArray() { free(ptr); }
  size_t ptr_as_int() {return (size_t)ptr; }
  scalar_t* ptr;
  size_t size;
};



void Fill(AlignedArray* out, scalar_t val) {
  /**
   * Fill the values of an aligned array with val
   */
  for (int i = 0; i < out->size; i++) {
    out->ptr[i] = val;
  }
}



// Decode a row-major logical index into the shared storage of a strided view.
inline size_t StridedIndex(size_t index, const std::vector<int32_t>& shape,
                           const std::vector<int32_t>& strides, size_t offset) {
  int64_t position = static_cast<int64_t>(offset);
  for (int axis = static_cast<int>(shape.size()) - 1; axis >= 0; --axis) {
    position += static_cast<int64_t>(index % shape[axis]) * strides[axis];
    index /= shape[axis];
  }
  return static_cast<size_t>(position);
}

void Compact(const AlignedArray& a, AlignedArray* out, std::vector<int32_t> shape,
             std::vector<int32_t> strides, size_t offset) {
  /**
   * Compact an array in memory
   *
   * Args:
   *   a: non-compact representation of the array, given as input
   *   out: compact version of the array to be written
   *   shape: shapes of each dimension for a and out
   *   strides: strides of the *a* array (not out, which has compact strides)
   *   offset: offset of the *a* array (not out, which has zero offset, being compact)
   *
   * Returns:
   *  void (you need to modify out directly, rather than returning anything; this is true for all the
   *  function will implement here, so we won't repeat this note.)
   */
  /// BEGIN SOLUTION
  for (size_t i = 0; i < out->size; ++i) {
    out->ptr[i] = a.ptr[StridedIndex(i, shape, strides, offset)];
  }
  /// END SOLUTION
}

void EwiseSetitem(const AlignedArray& a, AlignedArray* out, std::vector<int32_t> shape,
                  std::vector<int32_t> strides, size_t offset) {
  /**
   * Set items in a (non-compact) array
   *
   * Args:
   *   a: _compact_ array whose items will be written to out
   *   out: non-compact array whose items are to be written
   *   shape: shapes of each dimension for a and out
   *   strides: strides of the *out* array (not a, which has compact strides)
   *   offset: offset of the *out* array (not a, which has zero offset, being compact)
   */
  /// BEGIN SOLUTION
  for (size_t i = 0; i < a.size; ++i) {
    out->ptr[StridedIndex(i, shape, strides, offset)] = a.ptr[i];
  }
  /// END SOLUTION
}

void ScalarSetitem(const size_t size, scalar_t val, AlignedArray* out, std::vector<int32_t> shape,
                   std::vector<int32_t> strides, size_t offset) {
  /**
   * Set items is a (non-compact) array
   *
   * Args:
   *   size: number of elements to write in out array (note that this will note be the same as
   *         out.size, because out is a non-compact subset array);  it _will_ be the same as the
   *         product of items in shape, but convenient to just pass it here.
   *   val: scalar value to write to
   *   out: non-compact array whose items are to be written
   *   shape: shapes of each dimension of out
   *   strides: strides of the out array
   *   offset: offset of the out array
   */

  /// BEGIN SOLUTION
  for (size_t i = 0; i < size; ++i) {
    out->ptr[StridedIndex(i, shape, strides, offset)] = val;
  }
  /// END SOLUTION
}

void EwiseAdd(const AlignedArray& a, const AlignedArray& b, AlignedArray* out) {
  /**
   * Set entries in out to be the sum of correspondings entires in a and b.
   */
  for (size_t i = 0; i < a.size; i++) {
    out->ptr[i] = a.ptr[i] + b.ptr[i];
  }
}

void ScalarAdd(const AlignedArray& a, scalar_t val, AlignedArray* out) {
  /**
   * Set entries in out to be the sum of corresponding entry in a plus the scalar val.
   */
  for (size_t i = 0; i < a.size; i++) {
    out->ptr[i] = a.ptr[i] + val;
  }
}


/**
 * In the code the follows, use the above template to create analogous element-wise
 * and and scalar operators for the following functions.  See the numpy backend for
 * examples of how they should work.
 *   - EwiseMul, ScalarMul
 *   - EwiseDiv, ScalarDiv
 *   - ScalarPower
 *   - EwiseMaximum, ScalarMaximum
 *   - EwiseEq, ScalarEq
 *   - EwiseGe, ScalarGe
 *   - EwiseLog
 *   - EwiseExp
 *   - EwiseTanh
 *
 * If you implement all these naively, there will be a lot of repeated code, so
 * you are welcome (but not required), to use macros or templates to define these
 * functions (however you want to do so, as long as the functions match the proper)
 * signatures above.
 */


// Keep the elementwise and scalar versions of each operation consistent.
#define DEFINE_BINARY_OP(Name, expression)                                      \
  void Ewise##Name(const AlignedArray& a, const AlignedArray& b, AlignedArray* out) { \
    for (size_t i = 0; i < out->size; ++i) {                                    \
      const scalar_t x = a.ptr[i], y = b.ptr[i];                               \
      out->ptr[i] = (expression);                                              \
    }                                                                          \
  }                                                                            \
  void Scalar##Name(const AlignedArray& a, scalar_t y, AlignedArray* out) {       \
    for (size_t i = 0; i < out->size; ++i) {                                    \
      const scalar_t x = a.ptr[i];                                             \
      out->ptr[i] = (expression);                                              \
    }                                                                          \
  }

DEFINE_BINARY_OP(Mul, x * y)
DEFINE_BINARY_OP(Div, x / y)
DEFINE_BINARY_OP(Maximum, (std::isnan(x) || x > y) ? x : y)
DEFINE_BINARY_OP(Eq, x == y)
DEFINE_BINARY_OP(Ge, x >= y)
#undef DEFINE_BINARY_OP

void ScalarPower(const AlignedArray& a, scalar_t val, AlignedArray* out) {
  for (size_t i = 0; i < out->size; ++i) {
    out->ptr[i] = std::pow(a.ptr[i], val);
  }
}

#define DEFINE_UNARY_OP(Name, expression)                                      \
  void Ewise##Name(const AlignedArray& a, AlignedArray* out) {                   \
    for (size_t i = 0; i < out->size; ++i) {                                    \
      const scalar_t x = a.ptr[i];                                             \
      out->ptr[i] = (expression);                                              \
    }                                                                          \
  }

DEFINE_UNARY_OP(Log, std::log(x))
DEFINE_UNARY_OP(Exp, std::exp(x))
DEFINE_UNARY_OP(Tanh, std::tanh(x))
#undef DEFINE_UNARY_OP

void Matmul(const AlignedArray& a, const AlignedArray& b, AlignedArray* out, uint32_t m, uint32_t n,
            uint32_t p) {
  /**
   * Multiply two (compact) matrices into an output (also compact) matrix.  For this implementation
   * you can use the "naive" three-loop algorithm.
   *
   * Args:
   *   a: compact 2D array of size m x n
   *   b: compact 2D array of size n x p
   *   out: compact 2D array of size m x p to write the output to
   *   m: rows of a / out
   *   n: columns of a / rows of b
   *   p: columns of b / out
   */

  /// BEGIN SOLUTION
  std::fill(out->ptr, out->ptr + out->size, 0.0f);
  // Traversing a row of b in the inner loop gives contiguous memory accesses.
  for (size_t i = 0; i < m; ++i) {
    for (size_t k = 0; k < n; ++k) {
      const scalar_t value = a.ptr[i * n + k];
      for (size_t j = 0; j < p; ++j) {
        out->ptr[i * p + j] += value * b.ptr[k * p + j];
      }
    }
  }
  /// END SOLUTION
}

inline void AlignedDot(const float* __restrict__ a,
                       const float* __restrict__ b,
                       float* __restrict__ out) {

  /**
   * Multiply together two TILE x TILE matrices, and _add _the result to out (it is important to add
   * the result to the existing out, which you should not set to zero beforehand).  We are including
   * the compiler flags here that enable the compile to properly use vector operators to implement
   * this function.  Specifically, the __restrict__ keyword indicates to the compile that a, b, and
   * out don't have any overlapping memory (which is necessary in order for vector operations to be
   * equivalent to their non-vectorized counterparts (imagine what could happen otherwise if a, b,
   * and out had overlapping memory).  Similarly the __builtin_assume_aligned keyword tells the
   * compiler that the input array will be aligned to the appropriate blocks in memory, which also
   * helps the compiler vectorize the code.
   *
   * Args:
   *   a: compact 2D array of size TILE x TILE
   *   b: compact 2D array of size TILE x TILE
   *   out: compact 2D array of size TILE x TILE to write to
   */

  a = (const float*)__builtin_assume_aligned(a, TILE * ELEM_SIZE);
  b = (const float*)__builtin_assume_aligned(b, TILE * ELEM_SIZE);
  out = (float*)__builtin_assume_aligned(out, TILE * ELEM_SIZE);

  /// BEGIN SOLUTION
  for (size_t i = 0; i < TILE; ++i) {
    for (size_t k = 0; k < TILE; ++k) {
      const scalar_t value = a[i * TILE + k];
      for (size_t j = 0; j < TILE; ++j) {
        out[i * TILE + j] += value * b[k * TILE + j];
      }
    }
  }
  /// END SOLUTION
}

void MatmulTiled(const AlignedArray& a, const AlignedArray& b, AlignedArray* out, uint32_t m,
                 uint32_t n, uint32_t p) {
  /**
   * Matrix multiplication on tiled representations of array.  In this setting, a, b, and out
   * are all *4D* compact arrays of the appropriate size, e.g. a is an array of size
   *   a[m/TILE][n/TILE][TILE][TILE]
   * You should do the multiplication tile-by-tile to improve performance of the array (i.e., this
   * function should call `AlignedDot()` implemented above).
   *
   * Note that this function will only be called when m, n, p are all multiples of TILE, so you can
   * assume that this division happens without any remainder.
   *
   * Args:
   *   a: compact 4D array of size m/TILE x n/TILE x TILE x TILE
   *   b: compact 4D array of size n/TILE x p/TILE x TILE x TILE
   *   out: compact 4D array of size m/TILE x p/TILE x TILE x TILE to write to
   *   m: rows of a / out
   *   n: columns of a / rows of b
   *   p: columns of b / out
   *
   */
  /// BEGIN SOLUTION
  std::fill(out->ptr, out->ptr + out->size, 0.0f);
  const size_t rows = m / TILE, inner = n / TILE, cols = p / TILE;
  const size_t tile_elements = TILE * TILE;
  for (size_t i = 0; i < rows; ++i) {
    for (size_t j = 0; j < cols; ++j) {
      scalar_t* output_tile = out->ptr + (i * cols + j) * tile_elements;
      for (size_t k = 0; k < inner; ++k) {
        AlignedDot(a.ptr + (i * inner + k) * tile_elements,
                   b.ptr + (k * cols + j) * tile_elements, output_tile);
      }
    }
  }
  /// END SOLUTION
}

void ReduceMax(const AlignedArray& a, AlignedArray* out, size_t reduce_size) {
  /**
   * Reduce by taking maximum over `reduce_size` contiguous blocks.
   *
   * Args:
   *   a: compact array of size a.size = out.size * reduce_size to reduce over
   *   out: compact array to write into
   *   reduce_size: size of the dimension to reduce over
   */

  /// BEGIN SOLUTION
  if (reduce_size == 0) throw std::invalid_argument("maximum of an empty dimension");
  for (size_t i = 0; i < out->size; ++i) {
    scalar_t value = a.ptr[i * reduce_size];
    for (size_t j = 1; j < reduce_size; ++j) {
      const scalar_t next = a.ptr[i * reduce_size + j];
      value = (std::isnan(value) || value > next) ? value : next;
    }
    out->ptr[i] = value;
  }
  /// END SOLUTION
}

void ReduceSum(const AlignedArray& a, AlignedArray* out, size_t reduce_size) {
  /**
   * Reduce by taking sum over `reduce_size` contiguous blocks.
   *
   * Args:
   *   a: compact array of size a.size = out.size * reduce_size to reduce over
   *   out: compact array to write into
   *   reduce_size: size of the dimension to reduce over
   */

  /// BEGIN SOLUTION
  for (size_t i = 0; i < out->size; ++i) {
    scalar_t value = 0.0f;
    for (size_t j = 0; j < reduce_size; ++j) {
      value += a.ptr[i * reduce_size + j];
    }
    out->ptr[i] = value;
  }
  /// END SOLUTION
}

// Integer structure is owned by the pattern and is shared by forward and backward.
// Dense tensor buffers stay in Needle's existing CPU allocation type.
class CSRPattern {
 public:
  CSRPattern(int32_t rows, int32_t cols, std::vector<int32_t> offsets,
             std::vector<int32_t> indices)
      : rows_(rows), cols_(cols), offsets_(std::move(offsets)), indices_(std::move(indices)) {
    if (rows < 0 || cols < 0 || offsets_.size() != static_cast<size_t>(rows) + 1 ||
        offsets_.front() != 0 || offsets_.back() < 0 ||
        static_cast<size_t>(offsets_.back()) != indices_.size() ||
        !std::is_sorted(offsets_.begin(), offsets_.end())) {
      throw std::invalid_argument("invalid CSR row offsets");
    }
    for (int32_t col : indices_) {
      if (col < 0 || col >= cols) throw std::invalid_argument("CSR column index out of range");
    }
  }

  // Normal multiplication gathers feature rows. Its adjoint scatters output
  // gradient rows into columns of the original matrix, using the same weights.
  void Multiply(const AlignedArray& values, const AlignedArray& rhs, AlignedArray* out,
                size_t width, bool transpose) const {
    const size_t input_rows = transpose ? rows_ : cols_;
    const size_t output_rows = transpose ? cols_ : rows_;
    if (values.size != indices_.size() || rhs.size != input_rows * width ||
        out->size != output_rows * width) {
      throw std::invalid_argument("incompatible CSR multiplication buffers");
    }
    std::fill_n(out->ptr, out->size, 0.0f);
    for (int32_t row = 0; row < rows_; ++row) {
      for (int32_t e = offsets_[row]; e < offsets_[row + 1]; ++e) {
        const int32_t col = indices_[e];
        const size_t src = transpose ? row : col;
        const size_t dst = transpose ? col : row;
        for (size_t f = 0; f < width; ++f) {
          out->ptr[dst * width + f] += values.ptr[e] * rhs.ptr[src * width + f];
        }
      }
    }
  }

  // Only entries that exist in CSR have a trainable value.
  void ValueGradient(const AlignedArray& x, const AlignedArray& grad, AlignedArray* out,
                     size_t width) const {
    if (out->size != indices_.size() || x.size != static_cast<size_t>(cols_) * width ||
        grad.size != static_cast<size_t>(rows_) * width) {
      throw std::invalid_argument("incompatible CSR value-gradient buffers");
    }
    for (int32_t row = 0; row < rows_; ++row) {
      for (int32_t e = offsets_[row]; e < offsets_[row + 1]; ++e) {
        scalar_t dot = 0.0f;
        for (size_t f = 0; f < width; ++f) {
          dot += grad.ptr[static_cast<size_t>(row) * width + f]
               * x.ptr[static_cast<size_t>(indices_[e]) * width + f];
        }
        out->ptr[e] = dot;
      }
    }
  }

  // NumPy only gets read-only views. Keeping the owning Python object as their
  // base keeps these vectors alive even after the outer CSRMatrix is released.
  pybind11::array_t<int32_t> IndexView(bool offsets) const {
    const auto& a = offsets ? offsets_ : indices_;
    pybind11::array_t<int32_t> view(a.size(), a.data(),
        pybind11::cast(this, pybind11::return_value_policy::reference));
    view.attr("setflags")(false);
    return view;
  }

 private:
  int32_t rows_, cols_;
  std::vector<int32_t> offsets_, indices_;
};

}  // namespace cpu
}  // namespace needle

PYBIND11_MODULE(ndarray_backend_cpu, m) {
  namespace py = pybind11;
  using namespace needle;
  using namespace cpu;

  m.attr("__device_name__") = "cpu";
  m.attr("__tile_size__") = TILE;

  py::class_<AlignedArray>(m, "Array")
      .def(py::init<size_t>(), py::return_value_policy::take_ownership)
      .def("ptr", &AlignedArray::ptr_as_int)
      .def_readonly("size", &AlignedArray::size);

  // return numpy array (with copying for simplicity, otherwise garbage
  // collection is a pain)
  m.def("to_numpy", [](const AlignedArray& a, std::vector<size_t> shape,
                       std::vector<size_t> strides, size_t offset) {
    std::vector<size_t> numpy_strides = strides;
    std::transform(numpy_strides.begin(), numpy_strides.end(), numpy_strides.begin(),
                   [](size_t& c) { return c * ELEM_SIZE; });
    return py::array_t<scalar_t>(shape, numpy_strides, a.ptr + offset);
  });

  // convert from numpy (with copying)
  m.def("from_numpy", [](py::array_t<scalar_t> a, AlignedArray* out) {
    std::memcpy(out->ptr, a.request().ptr, out->size * ELEM_SIZE);
  });

  m.def("fill", Fill);
  m.def("compact", Compact);
  m.def("ewise_setitem", EwiseSetitem);
  m.def("scalar_setitem", ScalarSetitem);
  m.def("ewise_add", EwiseAdd);
  m.def("scalar_add", ScalarAdd);

  m.def("ewise_mul", EwiseMul);
  m.def("scalar_mul", ScalarMul);
  m.def("ewise_div", EwiseDiv);
  m.def("scalar_div", ScalarDiv);
  m.def("scalar_power", ScalarPower);

  m.def("ewise_maximum", EwiseMaximum);
  m.def("scalar_maximum", ScalarMaximum);
  m.def("ewise_eq", EwiseEq);
  m.def("scalar_eq", ScalarEq);
  m.def("ewise_ge", EwiseGe);
  m.def("scalar_ge", ScalarGe);

  m.def("ewise_log", EwiseLog);
  m.def("ewise_exp", EwiseExp);
  m.def("ewise_tanh", EwiseTanh);

  py::class_<CSRPattern>(m, "CSRPattern")
      .def(py::init<int32_t, int32_t, std::vector<int32_t>, std::vector<int32_t>>())
      .def_property_readonly("row_offsets", [](const CSRPattern& a) { return a.IndexView(true); })
      .def_property_readonly("col_indices", [](const CSRPattern& a) { return a.IndexView(false); })
      .def("multiply", &CSRPattern::Multiply)
      .def("value_gradient", &CSRPattern::ValueGradient);

  m.def("matmul", Matmul);
  m.def("matmul_tiled", MatmulTiled);

  m.def("reduce_max", ReduceMax);
  m.def("reduce_sum", ReduceSum);
}
