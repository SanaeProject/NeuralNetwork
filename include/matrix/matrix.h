#ifndef SANAE_NEURALNETWORK_MATRIX  
#define SANAE_NEURALNETWORK_MATRIX  

#include "../view/view.h"
#include <array>  
#include <execution>
#include <initializer_list>
#include <iosfwd>
#include <type_traits>  
#include <vector>  

// std::executionポリシー判定用の型
template<typename T> struct is_std_exec_policy : std::false_type {};
template<> struct is_std_exec_policy<std::execution::sequenced_policy>            : std::true_type {};
template<> struct is_std_exec_policy<std::execution::parallel_policy>             : std::true_type {};
template<> struct is_std_exec_policy<std::execution::parallel_unsequenced_policy> : std::true_type {};

// std::vectorまたはstd::array判定用の型
template<typename T> struct is_vector_or_array : std::false_type {};
template<typename T, typename Alloc> struct is_vector_or_array<std::vector<T, Alloc>> : std::true_type {};
template<typename T, std::size_t N>  struct is_vector_or_array<std::array<T, N>>      : std::true_type {};

// BLAS使用判定用の型
#ifdef USE_OPENBLAS
	template<typename T> struct can_use_blas : std::false_type {};
	template<> struct can_use_blas<float>  : std::true_type {};
	template<> struct can_use_blas<double> : std::true_type {};
#else
	template<typename T>
	struct can_use_blas : std::false_type {};
#endif

/**
* @brief 汎用的な行列クラスを提供します。
 * @tparam T 行列の要素型
 * @tparam RowMajor 行優先か列優先かを指定するブール値。デフォルトはtrue(行優先)。
 * @tparam Container 内部データコンテナの型。デフォルトはstd::vector<T>。
 * @tparam En Containerがstd::vectorまたはstd::arrayであることを保証するためのSFINAE用パラメータ。
 */
template<typename T, bool RowMajor = true, typename Container = std::vector<T>,
typename En = std::enable_if_t<is_vector_or_array<Container>::value>>
class Matrix {
protected:
	size_t _rows, _cols; /// 行数と列数
	Container _data; /// 内部データコンテナ

	/**
	 * @brief operationに従い二つの行列に対し演算を行います。
	 * @tparam execType 実行ポリシー(parallel_policy,parallel_unsequenced_policy,sequenced_policyから選択可能)
	 * @tparam calcType 加算減算などを行う関数ブジェクト
	 * @tparam TyCheck 実行ポリシーが上記のポリシーのどれかであるかを検証する
	 * @param to to = to operation other
	 * @param other to = to operation other
	 * @param execPolicy 実行ポリシー
	 * @param operation 関数オブジェクト
	 * @return void
	*/
	template<typename execType, typename calcType, typename TyCheck = std::enable_if_t<is_std_exec_policy<execType>::value>>
	void _calc(Container& to, const Container& other, execType execPolicy, calcType operation);
	/**
	 * @brief operationに従い行列とスカラーに対し演算を行います。
	 * @tparam execType 実行ポリシー(parallel_policy,parallel_unsequenced_policy,sequenced_policyから選択可能)
	 * @tparam calcType 加算減算などを行う関数ブジェクト
	 * @tparam TyCheck 実行ポリシーが上記のポリシーのどれかであるかを検証する
	 * @param to to = to operation other
	 * @param other to = to operation other
	 * @param execPolicy 実行ポリシー
	 * @param operation 関数オブジェクト
	 * @return void
	*/
	template<typename execType, typename calcType, typename TyCheck = std::enable_if_t<is_std_exec_policy<execType>::value>>
	void _calc(Container& to, const T& other, execType execPolicy, calcType operation);
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
	 * @brief 行列の行数と列数、初期値を指定して初期化するコンストラクタ
	 * @param rows 行数
	 * @param cols 列数
	 * @param initial 初期値
	 */
	Matrix(size_t rows, size_t cols, const T& initial);
	/**
	* @brief 行列の行数と列数、初期化関数を指定して初期化するコンストラクタ
	* @param rows 行数
	* @param cols 列数
	* @param func 初期化関数
	*/
	template<typename InitFunc,typename InitFuncCheck = std::enable_if_t<std::is_invocable_r_v<T, InitFunc>>>
	Matrix(size_t rows, size_t cols, InitFunc func);
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
	const Container& data() const;
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
	 * @brief BLASのGEMMを使用するかどうかを判定します。
	 * @return 使用する場合はtrue、使用しない場合はfalse
	 */
	bool is_blas_enabled() const;
	/**
	* @brief 行列の転置を行います。
	* @return 自身の参照
	*/
	Matrix& transpose();

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
	template<typename Ty, bool R, typename C, typename E>
	friend std::ostream& operator<<(std::ostream& os, const Matrix<Ty, R, C, E>& mat);

	// calc.hpp
	/**
	 * @brief 他の行列との加算を行います。
	 * @tparam use_blas BLASを使用するかどうか(デフォルトはfalse)
	 * @tparam execType 実行ポリシー(parallel_policy,parallel_unsequenced_policy,sequenced_policyから選択可能)デフォルトはstd::execution::parallel_unsequenced_policy
	 * @tparam TyCheck 実行ポリシーが上記のポリシーのどれかであるかを検証する
	 * @param other 加算する行列
	 * @param execPolicy 実行ポリシー(デフォルトはexecPolicy())
	 * @return 自身の参照
	 * @throws std::invalid_argument 行列の次元が一致しない場合
	 */
	template<bool use_blas = false, typename execType = std::execution::parallel_unsequenced_policy, typename TyCheck = std::enable_if_t<is_std_exec_policy<execType>::value>>
	Matrix& add(const Matrix& other, execType execPolicy = execType());
	/**
	 * @brief 他の行列との減算を行います。
	 * @tparam use_blas BLASを使用するかどうか(デフォルトはfalse)
	 * @tparam execType 実行ポリシー(parallel_policy,parallel_unsequenced_policy,sequenced_policyから選択可能)デフォルトはstd::execution::parallel_unsequenced_policy
	 * @tparam TyCheck 実行ポリシーが上記のポリシーのどれかであるかを検証する
	 * @param other 減算する行列
	 * @param execPolicy 実行ポリシー(デフォルトはexecPolicy())
	 * @return 自身の参照
	 * @throws std::invalid_argument 行列の次元が一致しない場合
	 */
	template<bool use_blas = false, typename execType = std::execution::parallel_unsequenced_policy, typename TyCheck = std::enable_if_t<is_std_exec_policy<execType>::value>>
	Matrix& sub(const Matrix& other, execType execPolicy = execType());
	/**
	 * @brief 他の行列とのアダマール積を行います。
	 * @tparam execType 実行ポリシー(parallel_policy,parallel_unsequenced_policy,sequenced_policyから選択可能)デフォルトはstd::execution::parallel_unsequenced_policy
	 * @tparam TyCheck 実行ポリシーが上記のポリシーのどれかであるかを検証する
	 * @param other 乗算する行列
	 * @param execPolicy 実行ポリシー(デフォルトはexecPolicy())
	 * @return 自身の参照
	 * @throws std::invalid_argument 行列の次元が一致しない場合
	 */
	template<typename execType = std::execution::parallel_unsequenced_policy, typename TyCheck = std::enable_if_t<is_std_exec_policy<execType>::value>>
	Matrix& hadamard_mul(const Matrix& other, execType execPolicy = execType());
	/**
	 * @brief スカラーとの乗算を行います。
	 * @tparam use_blas BLASを使用するかどうか(デフォルトはfalse)
	 * @tparam execType 実行ポリシー(parallel_policy,parallel_unsequenced_policy,sequenced_policyから選択可能)デフォルトはstd::execution::parallel_unsequenced_policy
	 * @tparam TyCheck 実行ポリシーが上記のポリシーのどれかであるかを検証する
	 * @param scalar 乗算するスカラー
	 * @param execPolicy 実行ポリシー(デフォルトはexecPolicy())
	 * @return 自身の参照
	 */
	template<bool use_blas = false, typename execType = std::execution::parallel_unsequenced_policy, typename TyCheck = std::enable_if_t<is_std_exec_policy<execType>::value>>
	Matrix& scalar_mul(const T& scalar, execType execPolicy = execType());
	/**
	 * @brief 他の行列とのアダマール除算を行います。
	 * @tparam execType 実行ポリシー(parallel_policy,parallel_unsequenced_policy,sequenced_policyから選択可能)デフォルトはstd::execution::parallel_unsequenced_policy
	 * @tparam TyCheck 実行ポリシーが上記のポリシーのどれかであるかを検証する
	 * @param other 除算する行列
	 * @param execPolicy 実行ポリシー(デフォルトはexecPolicy())
	 * @return 自身の参照
	 * @throws std::invalid_argument 行列の次元が一致しない場合、またはゼロ除算が発生した場合
	 */
	template<typename execType = std::execution::parallel_unsequenced_policy, typename TyCheck = std::enable_if_t<is_std_exec_policy<execType>::value>>
	Matrix& hadamard_div(const Matrix& other, execType execPolicy = execType());
	/**
	 * @brief スカラーとの除算を行います。
	 * @tparam execType 実行ポリシー(parallel_policy,parallel_unsequenced_policy,sequenced_policyから選択可能)デフォルトはstd::execution::parallel_unsequenced_policy
	 * @tparam TyCheck 実行ポリシーが上記のポリシーのどれかであるかを検証する
	 * @param scalar 除算するスカラー
	 * @param execPolicy 実行ポリシー(デフォルトはexecPolicy())
	 * @return 自身の参照
	 * @throws std::invalid_argument ゼロ除算が発生した場合
	 */
	template<typename execType = std::execution::parallel_unsequenced_policy, typename TyCheck = std::enable_if_t<is_std_exec_policy<execType>::value>>
	Matrix& scalar_div(const T& scalar, execType execPolicy = execType());
	/**
	 * @brief 他の行列との行列乗算を行います。
	 * @tparam use_blas BLASを使用するかどうか(デフォルトはfalse)
	 * @tparam OtherMajor 他の行列のメモリレイアウト
	 * @tparam MCheck RowMajorがfalseかつOtherMajorがtrueである場合にコンパイルエラーとする(効率が非常に悪いため)
	 * @param other 乗算する行列
	 * @return 自身の参照
	 * @throws std::invalid_argument 行列の次元が一致しない場合
	 */
	template<bool use_blas = false, bool OtherMajor, typename MCheck = std::enable_if_t<!(RowMajor == false && OtherMajor == true)>>
	Matrix& matrix_mul(const Matrix<T,OtherMajor>& other);
};

#endif // SANAE_NEURALNETWORK_MATRIX