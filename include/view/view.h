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
       using iterator_category = std::random_access_iterator_tag;  
       using value_type = T;  
       using difference_type = std::ptrdiff_t;  
       using pointer = T*;  
       using reference = T&;  

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
           : _data(data), _index(index), _stride(stride) {}  

       /**  
        * @brief イテレータのデリファレンス  
        * @return 現在の要素への参照  
        */  
       inline reference operator*() const { return _data[_index * _stride]; }  

       /**  
        * @brief イテレータのインクリメント（前置）  
        * @return 自身への参照  
        */  
       inline iterator& operator++() { ++_index; return *this; }  

       /**  
        * @brief イテレータのインクリメント（後置）  
        * @return インクリメント前のイテレータ  
        */  
       inline iterator operator++(int) { iterator tmp = *this; ++(*this); return tmp; }  

       // その他のメソッドも同様にコメントを追加可能  
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
template<typename T>
class View<const T> {
private:
    const T* _data;
    size_t _size;
    size_t _stride = 1;

public:
    View(const T* data, size_t size) : _data(data), _size(size) {}
    View(const T* data, size_t size, size_t stride) : _data(data), _size(size), _stride(stride) {}

    const T& operator[](size_t i) { return _data[i * _stride]; }
    const T& at(size_t i) const {
        if (i >= _size) throw std::out_of_range("Index out of range in View::at");
        return _data[i * _stride];
    }

    class iterator {
    public:
        using iterator_category = std::random_access_iterator_tag;
        using value_type = T;
        using difference_type = std::ptrdiff_t;
        using pointer = const T*;
        using reference = const T&;

    private:
        pointer _data;
        size_t _index;
        size_t _stride;

    public:
        inline iterator(pointer data, size_t index, size_t stride)
            : _data(data), _index(index), _stride(stride) {
        }

        inline reference operator*() const { return _data[_index * _stride]; }
        inline pointer operator->() const { return &_data[_index * _stride]; }
        inline iterator& operator++() { ++_index; return *this; }
        inline iterator operator++(int) { iterator tmp = *this; ++(*this); return tmp; }

        inline iterator& operator--() { --_index; return *this; }
        inline iterator operator--(int) { iterator tmp = *this; --(*this); return tmp; }

        inline iterator operator+(difference_type n) const { return iterator(_data, _index + n, _stride); }
        inline iterator operator-(difference_type n) const { return iterator(_data, _index - n, _stride); }
        inline difference_type operator-(const iterator& other) const { return _index - other._index; }

        inline bool operator==(const iterator& other) const { return _index == other._index; }
        inline bool operator!=(const iterator& other) const { return _index != other._index; }
        inline bool operator<(const iterator& other) const { return _index < other._index; }
        inline bool operator>(const iterator& other) const { return _index > other._index; }
        inline bool operator<=(const iterator& other) const { return _index <= other._index; }
        inline bool operator>=(const iterator& other) const { return _index >= other._index; }

        inline reference operator[](difference_type n) const { return _data[(_index + n) * _stride]; }
    };

    iterator begin() const { return iterator(_data, 0, _stride); }
    iterator end() const { return iterator(_data, _size, _stride); }

    size_t size() const { return _size; }
};

#endif