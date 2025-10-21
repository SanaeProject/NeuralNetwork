#ifndef SANAE_NEURALNETWORK_VIEW
#define SANAE_NEURALNETWORK_VIEW

#include <iterator>

/**  
* @class View  
* @brief 任意の型Tのデータに対するビューを提供するテンプレートクラス。  
*  
* このクラスは、データ配列に対してストライドアクセスを可能にし、  
* 範囲外アクセスを防ぐための安全なインターフェースを提供します。  
*  
* @tparam T データ型  
*/  
template<typename T>  
class View {
private:  
   T* _data; ///< データ配列へのポインタ  
   size_t _size; ///< データの要素数  
   size_t _stride = 1; ///< ストライド（デフォルトは1）  

public:  
   /**  
    * @brief コンストラクタ  
    * @param data データ配列へのポインタ  
    * @param size データの要素数  
    */  
   View(T* data, size_t size) : _data(data), _size(size) {}  

   /**  
    * @brief ストライドを指定するコンストラクタ  
    * @param data データ配列へのポインタ  
    * @param size データの要素数  
    * @param stride ストライド  
    */  
   View(T* data, size_t size, size_t stride) : _data(data), _size(size), _stride(stride) {}  

   /**  
    * @brief インデックス演算子  
    * @param i アクセスするインデックス  
    * @return 指定されたインデックスの要素への参照  
    */  
   T& operator[](size_t i) { return _data[i * _stride]; }  

   /**  
    * @brief 安全なインデックスアクセス  
    * @param i アクセスするインデックス  
    * @return 指定されたインデックスの要素への参照  
    * @throw std::out_of_range 範囲外アクセス時にスローされる例外  
    */  
   T& at(size_t i) {  
       if (i >= _size) throw std::out_of_range("Index out of range in View::at");  
       return _data[i * _stride];  
   }  

   /**  
    * @class iterator  
    * @brief Viewクラス用のランダムアクセスイテレータ  
    */  
   class iterator {
   public:
       using iterator_category = std::random_access_iterator_tag; ///< イテレータのカテゴリ
       using value_type = T; ///< イテレータの値の型
       using difference_type = std::ptrdiff_t; ///< イテレータの差の型
       using pointer = const T*; ///< イテレータのポインタ型
       using reference = const T&; ///< イテレータの参照型

   private:
       pointer _data; ///< データ配列へのポインタ
       size_t _index; ///< 現在のインデックス
       size_t _stride; ///< ストライド

   public:
        /**
        * @brief コンストラクタ
        * @param data データ配列へのポインタ
        * @param index 現在のインデックス
        * @param stride ストライド
        */
        inline iterator(pointer data, size_t index, size_t stride)
            : _data(data), _index(index), _stride(stride) {
        }
        /**
        * @brief デリファレンス演算子
        * @return 現在の要素への参照
        */
        inline reference operator*() const { return _data[_index * _stride]; }
        /**
        * @brief メンバアクセス演算子
        * @return 現在の要素へのポインタ
        */
        inline pointer operator->() const { return &_data[_index * _stride]; }
        /**
        * @brief 前置インクリメント演算子
        * @return インクリメント後のイテレータへの参照
        */
        inline iterator& operator++() { ++_index; return *this; }
        /**
         * @brief 後置インクリメント演算子
         * @return インクリメント前のイテレータのコピー
         */
        inline iterator operator++(int) { iterator tmp = *this; ++(*this); return tmp; }
        /**
         * @brief 前置デクリメント演算子
         * @return デクリメント後のイテレータへの参照
         */
        inline iterator& operator--() { --_index; return *this; }
        /**
        * @brief 後置デクリメント演算子
        * @return デクリメント前のイテレータのコピー
        */
        inline iterator operator--(int) { iterator tmp = *this; --(*this); return tmp; }
        /**
        * @brief イテレータの加算演算子
        * @param n 加算するオフセット
        * @return 新しいイテレータ
        */
        inline iterator operator+(difference_type n) const { return iterator(_data, _index + n, _stride); }
        /**
         * @brief イテレータの減算演算子
         * @param n 減算するオフセット
         * @return 新しいイテレータ
         */
        inline iterator operator-(difference_type n) const { return iterator(_data, _index - n, _stride); }
        /**
         * @brief イテレータ間の差を計算
         * @param other 比較する他のイテレータ
         * @return 差の値
         */
        inline difference_type operator-(const iterator& other) const { return _index - other._index; }
        /**
         * @brief イテレータの等価比較演算子
         * @param other 比較する他のイテレータ
         * @return 等価であればtrue、そうでなければfalse
         */
        inline bool operator==(const iterator& other) const { return _index == other._index; }
        /**
         * @brief イテレータの非等価比較演算子
         * @param other 比較する他のイテレータ
         * @return 非等価であればtrue、そうでなければfalse
         */
        inline bool operator!=(const iterator& other) const { return _index != other._index; }
        /**
         * @brief イテレータの小なり比較演算子
         * @param other 比較する他のイテレータ
         * @return 小なりであればtrue、そうでなければfalse
         */
        inline bool operator<(const iterator& other) const { return _index < other._index; }
        /**
         * @brief イテレータの大なり比較演算子
         * @param other 比較する他のイテレータ
         * @return 大なりであればtrue、そうでなければfalse
         */
        inline bool operator>(const iterator& other) const { return _index > other._index; }
        /**
         * @brief イテレータの小なりイコール比較演算子
         * @param other 比較する他のイテレータ
         * @return 小なりイコールであればtrue、そうでなければfalse
         */
        inline bool operator<=(const iterator& other) const { return _index <= other._index; }
        /**
         * @brief イテレータの大なりイコール比較演算子
         * @param other 比較する他のイテレータ
         * @return 大なりイコールであればtrue、そうでなければfalse
         */
        inline bool operator>=(const iterator& other) const { return _index >= other._index; }
        /**
         * @brief 添字演算子
         * @param n オフセット
         * @return 指定されたオフセットの要素への参照
         */
        inline reference operator[](difference_type n) const { return _data[(_index + n) * _stride]; }
   };

   /**  
    * @brief イテレータの開始位置を取得  
    * @return 開始位置のイテレータ  
    */  
   iterator begin() const { return iterator(_data, 0, _stride); }  

   /**  
    * @brief イテレータの終了位置を取得  
    * @return 終了位置のイテレータ  
    */  
   iterator end() const { return iterator(_data, _size, _stride); }  

   /**  
    * @brief データの要素数を取得  
    * @return データの要素数  
    */  
   size_t size() const { return _size; }  
};

#endif