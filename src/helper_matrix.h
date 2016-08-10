

#ifndef HELPER_MATRIX_H
#define HELPER_MATRIX_H

#include <array>
#include <cstring>
#include <functional>
#include <ostream>

namespace matrix_math {

template <typename T, size_t N, size_t M>
class matrix {
private:
	T data[N][M];
public:
	const size_t height = N;
	const size_t width = M;

	matrix() : width(M), height(N) {}
	matrix(const std::array<T, N*M> arr) : matrix<T, N, M>() {
		for (size_t n = 0; n < N; ++n) {
			for (size_t m = 0; m < M; ++m) {
				data[n][m] = arr[(n * M) + m];
			}
		}
	}



	// Getter
	T* operator[](std::size_t n) {
		return data[n];
	}
	const T* operator[](std::size_t n) const {
		return data[n];
	}
	std::array<T, N*M> to_array() {
		std::array<T, N*M> result;
		for (size_t n = 0; n < N; ++n) {
			for (size_t m = 0; m < M; ++m) {
				result[(n * M) + m] = (*this)[n][m];
			}
		}
		return result;
	}

	// Functions
	matrix<T, N, M>& apply(std::function<T(T)> func) {
		for (size_t n = 0; n < N; ++n) {
			for (size_t m = 0; m < M; ++m) {
				(*this)[n][m] = func((*this)[n][m]);
			}
		}
		return *this;
	}

	template <typename V>
	matrix<V, N, M> convert(std::function<V(T)> func) {
		matrix<V, N, M> out;
		for (size_t n = 0; n < N; ++n) {
			for (size_t m = 0; m < M; ++m) {
				out[n][m] = func((*this)[n][m]);
			}
		}
		return out;
	}

	matrix<T, N, M>& fill(const T value) {
		for (size_t n = 0; n < N; ++n) {
			for (size_t m = 0; m < M; ++m) {
				(*this)[n][m] = value;
			}
		}
		return *this;
	}


	// Operations
	template <typename V>
	matrix<T, N, M>& operator*=(const V& rhs) {
		for (size_t n = 0; n < N; ++n) {
			for (size_t m = 0; m < M; ++m) {
				(*this)[n][m] *= rhs;
			}
		}
		return *this;
	}

	template <typename V>
	friend matrix<T, N, M> operator*(matrix<T, N, M> lhs, const V& rhs) {
		for (size_t n = 0; n < N; ++n) {
			for (size_t m = 0; m < M; ++m) {
				lhs[n][m] *= rhs;
			}
		}
		return lhs;
	}

	friend std::array<T, N> operator*(const matrix<T, N, M> lhs, const std::array<T, M> vec) {
		std::array<T, N> out;
		for (size_t n = 0; n < N; ++n) {
			float temp = 0;
			for (size_t m = 0; m < M; ++m) {
				temp += lhs[n][m] * vec[m];
			}
			out[n] = temp;
		}
		return out;
	}

	template <size_t X>
	friend matrix<T, N, X> operator*(const matrix<T, N, M>& lhs, const matrix<T, M, X>& rhs) {
		matrix<T, N, X> out;
		for (size_t n = 0; n < N; ++n) {
			for (size_t x = 0; x < X; ++x) {
				T temp = 0;
				for (size_t m = 0; m < M; ++m) {
					temp += lhs[n][m] * rhs[m][x];
				}
				out[n][x] = temp;
			}
		}
		return out;
	}

	matrix<T, N, M>& operator*=(const matrix<T, M, M>& rhs) {
		matrix<T, N, M> temp = (*this) * rhs;
		std::memcpy(data, temp.data, sizeof(T) * N * M);
		return *this;
	}

	matrix<T, N, M>& operator+=(const matrix<T, N, M>& rhs) {
		for (size_t n = 0; n < N; ++n) {
			for (size_t m = 0; m < M; ++m) {
				(*this)[n][m] += rhs[n][m];
			}
		}
		return *this;
	}

	friend matrix<T, N, M> operator+(matrix<T, N, M> lhs, const matrix<T, N, M>& rhs) {
		lhs += rhs;
		return lhs;
	}

	matrix<T, N, M>& operator-=(const matrix<T, N, M>& rhs) {
		for (size_t n = 0; n < N; ++n) {
			for (size_t m = 0; m < M; ++m) {
				(*this)[n][m] -= rhs[n][m];
			}
		}
		return *this;
	}

	friend matrix<T, N, M> operator-(matrix<T, N, M> lhs, const matrix<T, N, M>& rhs) {
		lhs -= rhs;
		return lhs;
	}

	friend std::ostream& operator<<(std::ostream& os, const matrix<T, N, M>& matrix)	{
		for (size_t n = 0; n < N; ++n) {
			for (size_t m = 0; m < M; ++m) {
				if(m > 0) {
					os << ", ";
				}
				os << matrix[n][m];
			}
			os << std::endl;
		}
		return os;
	}

};


}

#endif /* HELPER_MATRIX_H */