#ifndef SANAE_NEURALNETWORK_MATRIX  
#define SANAE_NEURALNETWORK_MATRIX  

#include "../view/view.h"
#include <array>  
#include <execution>
#include <initializer_list>
#include <iosfwd>
#include <type_traits>  
#include <vector>  
#include <concepts>

// std::executionポリシー判定用の型
template<typename T>
concept StdExecPolicy = std::is_execution_policy_v<std::remove_cvref_t<T>>;

// std::vectorまたはstd::array判定用の型
template<typename T> struct is_vector_or_array : std::false_type {};
template<typename T, typename Alloc> struct is_vector_or_array<std::vector<T, Alloc>> : std::true_type {};
template<typename T, std::size_t N>  struct is_vector_or_array<std::array<T, N>>      : std::true_type {};
template<typename T> 
concept VectorOrArray = is_vector_or_array<std::remove_cvref_t<T>>::value;

// std::array判定用の型
template<typename T> struct is_std_array : std::false_type {};
template<typename T, std::size_t N>  struct is_std_array<std::array<T, N>> : std::true_type {};
template<typename T> 
concept StdArray = is_std_array<std::remove_cvref_t<T>>::value;

// BLAS使用判定用の型
template<typename T> struct can_use_blas : std::false_type {};
#if defined(USE_OPENBLAS)
// OpenBlas
	template<> struct can_use_blas<float>  : std::true_type {};
	template<> struct can_use_blas<double> : std::true_type {};
#elif defined(USE_CUBLAS)
// cuBLAS
	template<> struct can_use_blas<float> : std::true_type {};
	template<> struct can_use_blas<double> : std::true_type {};
#elif defined(USE_CLBLAST)
// clBLAST
	template<> struct can_use_blas<float> : std::true_type {};
	template<> struct can_use_blas<double> : std::true_type {};
#endif
template<typename T> concept CanUseBlas = can_use_blas<T>::value;

/**
 * @brief 汎用的な行列クラスを提供します。
 * @tparam T 行列の要素型
 * @tparam RowMajor 行優先か列優先かを指定するブール値。デフォルトはtrue(行優先)。
 * @tparam Container 内部データコンテナの型。デフォルトはstd::vector<T>。VectorOrArrayコンセプトを満たす必要があります。
 */
template<typename T, bool RowMajor = true, typename Container = std::vector<T>> requires VectorOrArray<Container>
class Matrix {
protected:
	size_t _rows, _cols; /// 行数と列数
	Container _data; /// 内部データコンテナ

	/**
	 * @brief operationに従い二つの行列に対し演算を行います。
	 * @tparam execType 実行ポリシー(parallel_policy,parallel_unsequenced_policy,sequenced_policyから選択可能)。StdExecPolicyコンセプトを満たす必要があります。
	 * @tparam calcType 加算減算などを行う関数オブジェクト
	 * @param to to = to operation other
	 * @param other to = to operation other
	 * @param execPolicy 実行ポリシー
	 * @param operation 関数オブジェクト
	 * @return void
	*/
	template<typename execType, typename calcType>
	void _calc(Container& to, const Container& other, execType execPolicy, calcType operation) const
		requires StdExecPolicy<execType>;

	/**
	 * @brief operationに従い行列とスカラーに対し演算を行います。
	 * @tparam execType 実行ポリシー(parallel_policy,parallel_unsequenced_policy,sequenced_policyから選択可能)。StdExecPolicyコンセプトを満たす必要があります。
	 * @tparam calcType 加算減算などを行う関数オブジェクト
	 * @param to to = to operation other
	 * @param other to = to operation other
	 * @param execPolicy 実行ポリシー
	 * @param operation 関数オブジェクト
	 * @return void
	*/
	template<typename execType, typename calcType>
	void _calc(Container& to, const T& other, execType execPolicy, calcType operation) const
		requires StdExecPolicy<execType>;
public:
	using Container2D = std::vector<std::vector<T>>;
	using InitContainer2D = std::initializer_list<std::initializer_list<T>>;

	// ctor.hpp
	/**
	 * @brief デフォルトコンストラクタ
	 */
	Matrix();

	/**
	 * @brief 行列の行数と列数を指定して初期化するコンストラクタ
	 * @param rows 行数
	 * @param cols 列数
	 */
	Matrix(size_t rows, size_t cols);

	/**
	 * @brief 行列の行数と列数を指定して初期化するコンストラクタ
	 * @param rows 行数
	 * @param cols 列数
	 * @param array 内部データコンテナの初期値を指定するコンテナ。少なくとも rows*cols 個の要素を保持している必要があります（std::array の場合は rows*cols が配列サイズ以下である必要があります）。
	 */
	Matrix(size_t rows, size_t cols, Container array);

	/**
	 * @brief 行列の行数と列数、初期化関数を指定して初期化するコンストラクタ
	 * @param rows 行数
	 * @param cols 列数
	 * @param func 初期化関数 (引数なしで呼び出せる関数オブジェクトで、返り値がT型に変換可能である必要があります)
	 * @param execPolicy 実行ポリシー (std::execution の実行ポリシーオブジェクト)。並列実行ポリシーを指定した場合、初期化関数 func は複数スレッドから並行して呼び出される可能性があるため、スレッドセーフである必要があります。
	 */
	template<typename InitFunc, typename ExecPolicy = std::execution::sequenced_policy>
	Matrix(size_t rows, size_t cols, InitFunc func, ExecPolicy execPolicy = ExecPolicy{})
	requires
		std::invocable<InitFunc> &&
		std::convertible_to<std::invoke_result_t<InitFunc>, T> &&
		StdExecPolicy<ExecPolicy>;

	/**
	 * @brief 2次元コンテナから初期化するコンストラクタ
	 * @param data 2次元コンテナ
	 */
	Matrix(const Container2D& data);

	/**
	 * @brief 2次元イニシャライザリストから初期化するコンストラクタ
	 * @param data 2次元イニシャライザリスト
	 */
	Matrix(const InitContainer2D& data);

	Matrix(const Matrix& other) = default;
	Matrix(Matrix&& other) noexcept = default;
	~Matrix() = default;

	// util.hpp
	/**
	 * @brief 行数を取得します。
	 * @return 行数
	 */
	size_t rows() const;

	/**
	 * @brief 列数を取得します。
	 * @return 列数
	 */
	size_t cols() const;

	/**
	 * @brief 内部データコンテナへの定数参照を取得します。
	 * @return 内部データコンテナへの定数参照
	 */
	const Container& data() const noexcept;

	/**
	 * @brief 行列のメモリレイアウトを変換します。
	 * @return メモリレイアウトが変換された新しい行列
	 */
	Matrix<T, !RowMajor> convertLayout() const;

	/**
	 * @brief 指定された行のビューを取得します。
	 * @param row 取得する行のインデックス
	 * @return 指定された行のビュー
	 */
	View<T> get_row(size_t row);

	/**
	 * @brief 指定された列のビューを取得します。
	 * @param col 取得する列のインデックス
	 * @return 指定された列のビュー
	 */
	View<T> get_col(size_t col);

	/**
	 * @brief 指定された行のポインタを取得します。
	 * @param row 取得する行のインデックス
	 * @return 指定された行のポインタ
	 * @note この関数は行優先レイアウトの場合にのみ有効で、列優先レイアウトの場合はコンパイルエラーになります。
	 */
	T* get_row_ptr(size_t row) requires RowMajor;

	/**
	 * @brief 指定された列のポインタを取得します。
	 * @param col 取得する列のインデックス
	 * @return 指定された列のポインタ
	 * @note この関数は列優先レイアウトの場合にのみ有効で、行優先レイアウトの場合はコンパイルエラーになります。
	 */
	T* get_col_ptr(size_t col) requires (!RowMajor);

	/**
	 * @brief 指定された行の定数ビューを取得します。(const版)
	 * @param row 取得する行のインデックス
	 * @return 指定された行の定数ビュー
	 */
	View<const T> get_row(size_t row) const;

	/**
	 * @brief 指定された列の定数ビューを取得します。(const版)
	 * @param col 取得する列のインデックス
	 * @return 指定された列の定数ビュー
	 */
	View<const T> get_col(size_t col) const;

	/**
	 * @brief 指定された行の定数ポインタを取得します。
	 * @param row 取得する行のインデックス
	 * @return 指定された行のポインタ
	 * @note この関数は行優先レイアウトの場合にのみ有効で、列優先レイアウトの場合はコンパイルエラーになります。
	 */
	const T* get_row_ptr(size_t row) const requires RowMajor;

	/**
	 * @brief 指定された列の定数ポインタを取得します。
	 * @param col 取得する列のインデックス
	 * @return 指定された列のポインタ
	 * @note この関数は列優先レイアウトの場合にのみ有効で、行優先レイアウトの場合はコンパイルエラーになります。
	 */
	const T* get_col_ptr(size_t col) const requires (!RowMajor);

	/**
	 * @brief BLASのGEMMを使用するかどうかを判定します。
	 * @return 使用する場合はtrue、使用しない場合はfalse
	 */
	bool is_blas_enabled() const;

	/**
	* @brief 行列の転置を行います。
	* @return 自身の参照
	*/
	Matrix& transpose();

	/**
	* @brief 行列の転置を行います。
	* @return 転置された新しい行列のコピー
	*/
	Matrix transpose_copy() const;
	
	/**
	 * @brief 行列の各要素に関数を適用します。
	 * @tparam Func 適用する関数の型
	 * @tparam ExecPolicy 使用する実行ポリシーの型（例：std::execution::sequenced_policy / parallel_policy など）
	 * @param func 各要素に適用する関数
	 * @param execPolicy 実行ポリシー。既定では逐次実行（sequenced）になり、並列ポリシーを指定した場合は
	 *                   関数funcがスレッドセーフであり、要素の処理順序に依存しないことが要求されます。
	 * @return 自身の参照
	 */
	template<typename Func, typename ExecPolicy = std::execution::sequenced_policy>
	Matrix& apply(Func func, ExecPolicy execPolicy = ExecPolicy{}) 
	requires
		std::invocable<Func, T> &&
		std::convertible_to<std::invoke_result_t<Func, T>, T> &&
		StdExecPolicy<ExecPolicy>;

	/**
	 * @brief 行列の各要素に関数を適用します。
	 * @tparam Func 適用する関数の型
	 * @tparam ExecPolicy 使用する実行ポリシーの型（例：std::execution::sequenced_policy / parallel_policy など）
	 * @param func 各要素に適用する関数
	 * @param execPolicy 実行ポリシー。既定では逐次実行（sequenced）になり、並列ポリシーを指定した場合は
	 *                   関数funcがスレッドセーフであり、要素の処理順序に依存しないことが要求されます。
	 * @return 新しい行列のコピー
	 */
	template<typename Func, typename ExecPolicy = std::execution::sequenced_policy>
	Matrix apply_copy(Func func, ExecPolicy execPolicy = ExecPolicy{}) const
	requires
		std::invocable<Func, T> &&
		std::convertible_to<std::invoke_result_t<Func, T>, T> &&
		StdExecPolicy<ExecPolicy>;

	/**
	 * @brief 行列の各行にdataを加算するなどの関数を適用します。
	 * @tparam CalcType 適用する関数の型
	 * @tparam ExecPolicy 使用する実行ポリシーの型（例：std::execution::sequenced_policy / parallel_policy など）
	 * @param operation 各要素に適用する関数 (a,b) -> a operation b の形で呼び出せる関数オブジェクトで、返り値がT型に変換可能である必要があります)
	 * @param execPolicy 実行ポリシー。既定では逐次実行（sequenced）になり、並列ポリシーを指定した場合は
	 *                   関数operationがスレッドセーフであり、要素の処理順序に依存しないことが要求されます。
	 * @return 自身の参照
	 */
	template<typename CalcType, typename ExecPolicy = std::execution::sequenced_policy>
	Matrix& apply_row(const Container& data, CalcType operation, ExecPolicy execPolicy = ExecPolicy{}) 
	requires
		StdExecPolicy<ExecPolicy>
		&& std::invocable<CalcType, T, T>
		&& std::convertible_to<std::invoke_result_t<CalcType, T, T>, T>;

	/**
	 * @brief 行列の各行にdataを加算するなどの関数を適用します。
	 * @tparam CalcType 適用する関数の型
	 * @tparam ExecPolicy 使用する実行ポリシーの型（例：std::execution::sequenced_policy / parallel_policy など）
	 * @param operation 各要素に適用する関数
	 * @param execPolicy 実行ポリシー。既定では逐次実行（sequenced）になり、並列ポリシーを指定した場合は
	 *                   関数operationがスレッドセーフであり、要素の処理順序に依存しないことが要求されます。
	 * @return 新しい行列のコピー
	 */
	template<typename CalcType, typename ExecPolicy = std::execution::sequenced_policy>
	Matrix apply_row_copy(const Container& data, CalcType operation, ExecPolicy execPolicy = ExecPolicy{}) const
	requires
		StdExecPolicy<ExecPolicy>
		&& std::invocable<CalcType, T, T>
		&& std::convertible_to<std::invoke_result_t<CalcType, T, T>, T>;

	// ops.hpp
	/**
	 * @brief 行列の要素にアクセスするための演算子を定義します。
	 * @param row 行インデックス
	 * @param col 列インデックス
	 * @return 指定された位置の要素への参照
	 */
	T& operator()(size_t row, size_t col);

	/**
	 * @brief 行列の要素にアクセスするための演算子を定義します。
	 * @param index 1次元インデックス
	 * @return 指定された位置の要素への参照
	 * @note 1次元インデックスは行優先または列優先のメモリレイアウトに基づいて解釈されます。
	 */
	T& operator()(size_t index);

	/**
	 * @brief 行列の要素にアクセスするための演算子を定義します。
	 * @param index 1次元インデックス
	 * @return 指定された位置の要素への参照
	 * @note 1次元インデックスは行優先または列優先のメモリレイアウトに基づいて解釈されます。
	 */
	T& operator[](size_t index);

	/**
	 * @brief 行列の要素にアクセスするための定数演算子を定義します。
	 * @param row 行インデックス
	 * @param col 列インデックス
	 * @return 指定された位置の要素への定数参照
	 */
	const T& operator()(size_t row, size_t col) const;

	/**
	 * @brief 行列の要素にアクセスするための定数演算子を定義します。
	 * @param index 1次元インデックス
	 * @return 指定された位置の要素への定数参照
	 * @note 1次元インデックスは行優先または列優先のメモリレイアウトに基づいて解釈されます。
	 */
	const T& operator()(size_t index) const;

	/**
	 * @brief 行列の要素にアクセスするための定数演算子を定義します。
	 * @param index 1次元インデックス
	 * @return 指定された位置の要素への定数参照
	 * @note 1次元インデックスは行優先または列優先のメモリレイアウトに基づいて解釈されます。
	 */
	const T& operator[](size_t index) const;

	/**
	 * @brief 他の行列との等価比較を行います。
	 * @param other 比較する行列
	 * @return 等価であればtrue、そうでなければfalse
	 */
	bool operator==(const Matrix& other) const;

	/**
	 * @brief 他の行列との非等価比較を行います。
	 * @param other 比較する行列
	 * @return 非等価であればtrue、そうでなければfalse
	 */
	bool operator!=(const Matrix& other) const;

	/**
	 * @brief 他の行列を代入します。
	 * @param other 代入する行列
	 * @return 自身の参照
	 */
	Matrix& operator=(const Matrix& other) = default;

	/**
	 * @brief 他の行列を加算します。
	 * @param other 加算する行列
	 * @return 新しい行列のコピー
	 */
	Matrix operator+(const Matrix& other) const;

	/**
	 * @brief 他の行列を加算します。
	 * @param scalar 加算するスカラー
	 * @return 新しい行列のコピー
	 */
	Matrix operator+(const T& scalar) const;

	/**
	 * @brief 他の行列を減算します。
	 * @param other 減算する行列
	 * @return 新しい行列のコピー
	 */
	Matrix operator-(const Matrix& other) const;

	/**
	 * @brief 他の行列を減算します。
	 * @param scalar 減算するスカラー
	 * @return 新しい行列のコピー
	 */
	Matrix operator-(const T& scalar) const;

	/**
	 * @brief 行列積を行います。
	 * @param other 乗算する行列
	 * @return 新しい行列のコピー
	 */
	Matrix operator*(const Matrix& other) const;

	/**
	 * @brief 行列積を行います。
	 * @param scalar 乗算するスカラー
	 * @return 新しい行列のコピー
	 */
	Matrix operator*(const T& scalar) const;

	/**
	 * @brief アダマール積を行います。
	 * @param other 乗算する行列
	 * @return 新しい行列のコピー
	 */
	Matrix operator^(const Matrix& other) const;

	/**
	 * @brief 行列の要素ごとの除算を行います。
	 * @param other 除算する行列
	 * @return 新しい行列のコピー
	 */
	Matrix operator/(const Matrix& other) const;

	/**
	 * @brief 行列の要素ごとの除算を行います。
	 * @param scalar 除算するスカラー
	 * @return 新しい行列のコピー
	 */
	Matrix operator/(const T& scalar) const;

	/**
	 * @brief 他の行列との等価比較を行います。(メモリレイアウトが異なる場合)
	 * @param other 比較する行列
	 * @return 等価であればtrue、そうでなければfalse
	 * @note メモリレイアウトが異なる場合低速になる可能性があります。
	 */
	bool operator==(const Matrix<T, !RowMajor>& other) const;

	/**
	 * @brief 他の行列との非等価比較を行います。(メモリレイアウトが異なる場合)
	 * @param other 比較する行列
	 * @return 非等価であればtrue、そうでなければfalse
	 * @note メモリレイアウトが異なる場合低速になる可能性があります。
	 */
	bool operator!=(const Matrix<T, !RowMajor>& other) const;

	/**
	 * @brief 行列を標準出力ストリームに出力するための演算子を定義します。
	 * @param os 出力ストリーム
	 * @param mat 出力する行列
	 * @return 出力ストリームへの参照
	 */
	template<typename Ty, bool R, typename C>
	friend std::ostream& operator<<(std::ostream&, const Matrix<Ty, R, C>&);

	// calc.hpp
	/**
	 * @brief 他の行列との加算を行います。
	 * @tparam use_blas BLASを使用するかどうか(デフォルトはfalse)
	 * @tparam execType 実行ポリシー(parallel_policy,parallel_unsequenced_policy,sequenced_policyから選択可能)デフォルトはstd::execution::sequenced_policy。StdExecPolicyコンセプトを満たす必要があります。
	 * @param other 加算する行列
	 * @param execPolicy 実行ポリシー(デフォルトはexecPolicy())
	 * @return 自身の参照
	 * @throws std::invalid_argument 行列の次元が一致しない場合
	 */
	template<bool use_blas = false, typename execType = std::execution::sequenced_policy>
	Matrix& add(const Matrix& other, execType execPolicy = execType()) requires StdExecPolicy<execType>;

	/**
	 * @brief 他の行列との加算を行います。
	 * @tparam use_blas BLASを使用するかどうか(デフォルトはfalse)
	 * @tparam execType 実行ポリシー(parallel_policy,parallel_unsequenced_policy,sequenced_policyから選択可能)デフォルトはstd::execution::sequenced_policy。StdExecPolicyコンセプトを満たす必要があります。
	 * @param other 加算する行列
	* @param execPolicy 実行ポリシー(デフォルトはexecPolicy())
	* @return 新しい行列のコピー
	 * @throws std::invalid_argument 行列の次元が一致しない場合
	 */
	template<bool use_blas = false, typename execType = std::execution::sequenced_policy>
	Matrix add_copy(const Matrix& other, execType execPolicy = execType()) const requires StdExecPolicy<execType>;

	/**
	 * @brief 他の行列との減算を行います。
	 * @tparam use_blas BLASを使用するかどうか(デフォルトはfalse)
	 * @tparam execType 実行ポリシー(parallel_policy,parallel_unsequenced_policy,sequenced_policyから選択可能)デフォルトはstd::execution::sequenced_policy。StdExecPolicyコンセプトを満たす必要があります。
	 * @param other 減算する行列
	 * @param execPolicy 実行ポリシー(デフォルトはexecPolicy())
	 * @return 自身の参照
	 * @throws std::invalid_argument 行列の次元が一致しない場合
	 */
	template<bool use_blas = false, typename execType = std::execution::sequenced_policy>
	Matrix& sub(const Matrix& other, execType execPolicy = execType()) requires StdExecPolicy<execType>;

	/**
	 * @brief 他の行列との減算を行います。
	 * @tparam use_blas BLASを使用するかどうか(デフォルトはfalse)
	 * @tparam execType 実行ポリシー(parallel_policy,parallel_unsequenced_policy,sequenced_policyから選択可能)デフォルトはstd::execution::sequenced_policy。StdExecPolicyコンセプトを満たす必要があります。
	 * @param other 減算する行列
	 * @param execPolicy 実行ポリシー(デフォルトはexecPolicy())
	 * @return 新しい行列のコピー
	 * @throws std::invalid_argument 行列の次元が一致しない場合
	 */
	template<bool use_blas = false, typename execType = std::execution::sequenced_policy>
	Matrix sub_copy(const Matrix& other, execType execPolicy = execType()) const requires StdExecPolicy<execType>;

	/**
	 * @brief 他の行列とのアダマール積を行います。
	 * @tparam execType 実行ポリシー(parallel_policy,parallel_unsequenced_policy,sequenced_policyから選択可能)デフォルトはstd::execution::sequenced_policy。StdExecPolicyコンセプトを満たす必要があります。
	 * @param other 乗算する行列
	 * @param execPolicy 実行ポリシー(デフォルトはexecPolicy())
	 * @return 自身の参照
	 * @throws std::invalid_argument 行列の次元が一致しない場合
	 */
	template<typename execType = std::execution::sequenced_policy>
	Matrix& hadamard_mul(const Matrix& other, execType execPolicy = execType()) requires StdExecPolicy<execType>;

	/**
	 * @brief 他の行列とのアダマール積を行います。
	 * @tparam execType 実行ポリシー(parallel_policy,parallel_unsequenced_policy,sequenced_policyから選択可能)デフォルトはstd::execution::sequenced_policy。StdExecPolicyコンセプトを満たす必要があります。
	 * @param other 乗算する行列
	 * @param execPolicy 実行ポリシー(デフォルトはexecPolicy())
	 * @return 新しい行列のコピー
	 * @throws std::invalid_argument 行列の次元が一致しない場合
	 */
	template<typename execType = std::execution::sequenced_policy>
	Matrix hadamard_mul_copy(const Matrix& other, execType execPolicy = execType()) const requires StdExecPolicy<execType>;

	/**
	 * @brief スカラーとの乗算を行います。
	 * @tparam use_blas BLASを使用するかどうか(デフォルトはfalse)
	 * @tparam execType 実行ポリシー(parallel_policy,parallel_unsequenced_policy,sequenced_policyから選択可能)デフォルトはstd::execution::sequenced_policy。StdExecPolicyコンセプトを満たす必要があります。
	 * @param scalar 乗算するスカラー
	 * @param execPolicy 実行ポリシー(デフォルトはexecPolicy())
	 * @return 自身の参照
	 */
	template<bool use_blas = false, typename execType = std::execution::sequenced_policy>
	Matrix& scalar_mul(const T& scalar, execType execPolicy = execType()) requires StdExecPolicy<execType>;

	/**
	 * @brief スカラーとの乗算を行います。
	 * @tparam use_blas BLASを使用するかどうか(デフォルトはfalse)
	 * @tparam execType 実行ポリシー(parallel_policy,parallel_unsequenced_policy,sequenced_policyから選択可能)デフォルトはstd::execution::sequenced_policy。StdExecPolicyコンセプトを満たす必要があります。
	 * @param scalar 乗算するスカラー
	 * @param execPolicy 実行ポリシー(デフォルトはexecPolicy())
	 * @return 新しい行列のコピー
	 */
	template<bool use_blas = false, typename execType = std::execution::sequenced_policy>
	Matrix scalar_mul_copy(const T& scalar, execType execPolicy = execType()) const requires StdExecPolicy<execType>;

	/**
	 * @brief 他の行列とのアダマール除算を行います。
	 * @tparam execType 実行ポリシー(parallel_policy,parallel_unsequenced_policy,sequenced_policyから選択可能)デフォルトはstd::execution::sequenced_policy。StdExecPolicyコンセプトを満たす必要があります。
	 * @param other 除算する行列
	 * @param execPolicy 実行ポリシー(デフォルトはexecPolicy())
	 * @return 自身の参照
	 * @throws std::invalid_argument 行列の次元が一致しない場合、またはゼロ除算が発生した場合
	 */
	template<typename execType = std::execution::sequenced_policy>
	Matrix& hadamard_div(const Matrix& other, execType execPolicy = execType()) requires StdExecPolicy<execType>;

	/**
	 * @brief 他の行列とのアダマール除算を行います。
	 * @tparam execType 実行ポリシー(parallel_policy,parallel_unsequenced_policy,sequenced_policyから選択可能)デフォルトはstd::execution::sequenced_policy。StdExecPolicyコンセプトを満たす必要があります。
	 * @param other 除算する行列
	 * @param execPolicy 実行ポリシー(デフォルトはexecPolicy())
	 * @return 新しい行列のコピー
	 * @throws std::invalid_argument 行列の次元が一致しない場合、またはゼロ除算が発生した場合
	 */
	template<typename execType = std::execution::sequenced_policy>
	Matrix hadamard_div_copy(const Matrix& other, execType execPolicy = execType()) const requires StdExecPolicy<execType>;

	/**
	 * @brief スカラーとの除算を行います。
	 * @tparam execType 実行ポリシー(parallel_policy,parallel_unsequenced_policy,sequenced_policyから選択可能)デフォルトはstd::execution::sequenced_policy。StdExecPolicyコンセプトを満たす必要があります。
	 * @param scalar 除算するスカラー
	 * @param execPolicy 実行ポリシー(デフォルトはexecPolicy())
	 * @return 自身の参照
	 * @throws std::invalid_argument ゼロ除算が発生した場合
	 */
	template<typename execType = std::execution::sequenced_policy>
	Matrix& scalar_div(const T& scalar, execType execPolicy = execType()) requires StdExecPolicy<execType>;

	/**
	 * @brief スカラーとの除算を行います。
	 * @tparam execType 実行ポリシー(parallel_policy,parallel_unsequenced_policy,sequenced_policyから選択可能)デフォルトはstd::execution::sequenced_policy。StdExecPolicyコンセプトを満たす必要があります。
	 * @param scalar 除算するスカラー
	 * @param execPolicy 実行ポリシー(デフォルトはexecPolicy())
	 * @return 新しい行列のコピー
	 * @throws std::invalid_argument ゼロ除算が発生した場合
	 */
	template<typename execType = std::execution::sequenced_policy>
	Matrix scalar_div_copy(const T& scalar, execType execPolicy = execType{}) const requires StdExecPolicy<execType>;

	/**
	 * @brief 各列の和を計算します。{{1,2,3},{4,5,6}} -> {{5,7,9}}
	 * @return 新しい行列のコピー
	 * @note 結果の行列は1行cols列の行列になります。
	 * @note execPolicyは実行ポリシーを指定します。列優先レイアウトの場合のみ有効です。
	 */
	template<typename execType = std::execution::sequenced_policy>
	Matrix sum_rows(execType execPolicy = execType{}) const requires StdExecPolicy<execType>;

	/**
	 * @brief 列の和を計算します。{{1,2,3},{4,5,6}} -> {{6},{15}}
	 * @return 新しい行列のコピー
	 * @note execPolicyは実行ポリシーを指定します。列優先レイアウトの場合のみ有効です。
	 */
	template<typename execType = std::execution::sequenced_policy>
	Matrix sum_cols(execType execPolicy = execType{}) const requires StdExecPolicy<execType>;

	/**
	 * @brief 他の行列との行列乗算を行います。
	 * @tparam use_blas BLASを使用するかどうか(デフォルトはfalse)
	 * @tparam OtherMajor 他の行列のメモリレイアウト
	 * @tparam MCheck RowMajorがfalseかつOtherMajorがtrueである場合にコンパイルエラーとする(効率が非常に悪いため)
	 * @param other 乗算する行列
	 * @return 自身の参照
	 * @throws std::invalid_argument 行列の次元が一致しない場合
	 */
	template<bool use_blas = false, bool OtherMajor, typename OtherContainer>
	inline Matrix& matrix_mul(const Matrix<T, OtherMajor, OtherContainer>& other)
	requires (!(RowMajor == false && OtherMajor == true));

	/**
	 * @brief 他の行列との行列乗算を行います。
	 * @tparam use_blas BLASを使用するかどうか(デフォルトはfalse)
	 * @tparam OtherMajor 他の行列のメモリレイアウト
	 * @tparam MCheck RowMajorがfalseかつOtherMajorがtrueである場合にコンパイルエラーとする(効率が非常に悪いため)
	 * @param other 乗算する行列
	 * @return 新しい行列のコピー
	 * @throws std::invalid_argument 行列の次元が一致しない場合
	 */
	template<bool use_blas = false, bool OtherMajor, typename OtherContainer>
	inline Matrix matrix_mul_copy(const Matrix<T, OtherMajor, OtherContainer>& other) const
	requires (!(RowMajor == false && OtherMajor == true));
};

#endif // SANAE_NEURALNETWORK_MATRIX