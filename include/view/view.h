#ifndef SANAE_NEURALNETWORK_VIEW
#define SANAE_NEURALNETWORK_VIEW

#include <iterator>

/**  
* @class View  
* @brief �C�ӂ̌^T�̃f�[�^�ɑ΂���r���[��񋟂���e���v���[�g�N���X�B  
*  
* ���̃N���X�́A�f�[�^�z��ɑ΂��ăX�g���C�h�A�N�Z�X���\�ɂ��A  
* �͈͊O�A�N�Z�X��h�����߂̈��S�ȃC���^�[�t�F�[�X��񋟂��܂��B  
*  
* @tparam T �f�[�^�^  
*/  
template<typename T>  
class View {  
private:  
   T* _data; ///< �f�[�^�z��ւ̃|�C���^  
   size_t _size; ///< �f�[�^�̗v�f��  
   size_t _stride = 1; ///< �X�g���C�h�i�f�t�H���g��1�j  

public:  
   /**  
    * @brief �R���X�g���N�^  
    * @param data �f�[�^�z��ւ̃|�C���^  
    * @param size �f�[�^�̗v�f��  
    */  
   View(T* data, size_t size) : _data(data), _size(size) {}  

   /**  
    * @brief �X�g���C�h���w�肷��R���X�g���N�^  
    * @param data �f�[�^�z��ւ̃|�C���^  
    * @param size �f�[�^�̗v�f��  
    * @param stride �X�g���C�h  
    */  
   View(T* data, size_t size, size_t stride) : _data(data), _size(size), _stride(stride) {}  

   /**  
    * @brief �C���f�b�N�X���Z�q  
    * @param i �A�N�Z�X����C���f�b�N�X  
    * @return �w�肳�ꂽ�C���f�b�N�X�̗v�f�ւ̎Q��  
    */  
   T& operator[](size_t i) { return _data[i * _stride]; }  

   /**  
    * @brief ���S�ȃC���f�b�N�X�A�N�Z�X  
    * @param i �A�N�Z�X����C���f�b�N�X  
    * @return �w�肳�ꂽ�C���f�b�N�X�̗v�f�ւ̎Q��  
    * @throw std::out_of_range �͈͊O�A�N�Z�X���ɃX���[������O  
    */  
   T& at(size_t i) {  
       if (i >= _size) throw std::out_of_range("Index out of range in View::at");  
       return _data[i * _stride];  
   }  

   /**  
    * @class iterator  
    * @brief View�N���X�p�̃����_���A�N�Z�X�C�e���[�^  
    */  
   class iterator {  
   public:  
       using iterator_category = std::random_access_iterator_tag;  
       using value_type = T;  
       using difference_type = std::ptrdiff_t;  
       using pointer = T*;  
       using reference = T&;  

   private:  
       pointer _data; ///< �f�[�^�z��ւ̃|�C���^  
       size_t _index; ///< ���݂̃C���f�b�N�X  
       size_t _stride; ///< �X�g���C�h  

   public:  
       /**  
        * @brief �R���X�g���N�^  
        * @param data �f�[�^�z��ւ̃|�C���^  
        * @param index ���݂̃C���f�b�N�X  
        * @param stride �X�g���C�h  
        */  
       inline iterator(pointer data, size_t index, size_t stride)  
           : _data(data), _index(index), _stride(stride) {}  

       /**  
        * @brief �C�e���[�^�̃f���t�@�����X  
        * @return ���݂̗v�f�ւ̎Q��  
        */  
       inline reference operator*() const { return _data[_index * _stride]; }  

       /**  
        * @brief �C�e���[�^�̃C���N�������g�i�O�u�j  
        * @return ���g�ւ̎Q��  
        */  
       inline iterator& operator++() { ++_index; return *this; }  

       /**  
        * @brief �C�e���[�^�̃C���N�������g�i��u�j  
        * @return �C���N�������g�O�̃C�e���[�^  
        */  
       inline iterator operator++(int) { iterator tmp = *this; ++(*this); return tmp; }  

       // ���̑��̃��\�b�h�����l�ɃR�����g��ǉ��\  
   };  

   /**  
    * @brief �C�e���[�^�̊J�n�ʒu���擾  
    * @return �J�n�ʒu�̃C�e���[�^  
    */  
   iterator begin() const { return iterator(_data, 0, _stride); }  

   /**  
    * @brief �C�e���[�^�̏I���ʒu���擾  
    * @return �I���ʒu�̃C�e���[�^  
    */  
   iterator end() const { return iterator(_data, _size, _stride); }  

   /**  
    * @brief �f�[�^�̗v�f�����擾  
    * @return �f�[�^�̗v�f��  
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