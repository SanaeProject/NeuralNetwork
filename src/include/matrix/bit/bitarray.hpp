#include <cstddef>
#include <cstdint>
#include <type_traits>
#include <vector>
#include <cstdint>
#include <stdexcept>
#include <random>
#include <bitset>

template<typename PackType>
concept PackTypeConcept = std::is_integral_v<PackType> && std::is_unsigned_v<PackType>;

template<typename Container>
concept ContainerConcept = requires(Container c, size_t idx) {
    { c.at(idx) } -> std::same_as<typename Container::value_type&>;
    { c.size() } -> std::convertible_to<std::size_t>;
    { c.max_size() } -> std::convertible_to<std::size_t>;
    c.begin();
    c.end();
};
template<typename Container, typename ValueType>
concept ResizableContainerConcept = requires(Container c, size_t new_size, ValueType val) {
     c.resize(new_size);
     c.assign(new_size, val);
};

template<typename PackType = uint8_t, typename Container = std::vector<PackType>>
requires PackTypeConcept<PackType> && ContainerConcept<Container>
class BitArray {
private:
    static constexpr size_t _PACKBITS = sizeof(PackType) * 8; //< 1パックあたりのビット数
    Container _data;
    size_t _bits;
    
    /**
     * @brief 指定されたビット数に必要な要素数(パック数)を計算します。
     * @param bits ビット数
     * @return 必要な要素数
     */
    static size_t _get_pack_size(size_t bits) {
        return (bits + _PACKBITS - 1) / _PACKBITS;
    }
    
    /**
     * @brief 指定された要素数に相当するビット数を計算します。
     * @param packs パック数
     * @return ビット数
     */
    static size_t _get_bits_size(const size_t& packs) {
        return packs * _PACKBITS;
    }

    /**
     * @brief 指定された位置のビットの値を取得します。
     * @param n ビットインデックス
     * @return 指定された位置のビットの値
     * @throws std::out_of_range ビットインデックスが範囲外の場合
     */
    bool _get_bit(size_t n) const {
        if (n >= _bits) throw std::out_of_range("not enough bits to get");
        const size_t pack_idx = n / _PACKBITS;
        const size_t bit_idx = _PACKBITS - 1 - n % _PACKBITS;
        return (this->_data[pack_idx] >> bit_idx) & 1;
    }

    /**
     * @brief 指定された位置のビットを設定します。
     * @param n ビットインデックス
     * @param val 設定する値
     * @throws std::out_of_range ビットインデックスが範囲外の場合
     */
    void _set_bit(size_t n, bool val) {
        if (n >= this->_bits) throw std::out_of_range("not enough bits to set");
        const size_t pack_idx = n / _PACKBITS;
        const size_t bit_idx = _PACKBITS - 1 - n % _PACKBITS;
        if (val) {
            this->_data[pack_idx] |= (PackType(1) << bit_idx);
        } else {
            this->_data[pack_idx] &= ~(PackType(1) << bit_idx);
        }
    }

public:
    // constructors
    BitArray();
    explicit BitArray(const Container& a): _data(a), _bits(_get_bits_size(a.size()))
    {};

    /**
     * @brief BitArrayを指定されたビット数で初期化します。
     * @param bits ビット数
     */
    explicit BitArray(const size_t& bits): _bits(bits){
        size_t req_size = _get_pack_size(bits);

        if(_data.max_size() < req_size)
            throw std::runtime_error("Container max_size is too small for the specified number of bits.");

        if constexpr (ResizableContainerConcept<Container, bool>) {
            _data.assign(req_size, static_cast<PackType>(false));
        } else {
            std::fill(_data.begin(), _data.end(), static_cast<PackType>(false));
        }
    }

    /**
     * @brief BitArrayを指定されたビット数と初期値で初期化します。
     * @param bits ビット数
     * @param init 初期値
     */
    explicit BitArray(const size_t& bits, const bool& init): _bits(bits){
        PackType init_val = static_cast<PackType>(0);
        size_t req_size = _get_pack_size(bits);
        
        if(init)
            init_val = ~init_val;

        if(this->_data.max_size() < req_size)
            throw std::runtime_error("Container max_size is too small for the specified number of bits."); 

        if constexpr (ResizableContainerConcept<Container, PackType>) {
            _data.assign(req_size, init_val);
        } else {
            std::fill(_data.begin(), _data.end(), init_val);
        }
    }
    auto operator<=>(const BitArray&) const = default;

    bool operator [](const size_t& bits){
        return this->_get_bit(bits);
    }
    bool operator ()(const size_t& bits){
        return this->_get_bit(bits);
    }

    /**
     * @brief BitArrayのサイズを変更します。
     * @param bits 新しいビット数
     * @return 自身の参照
     */
    BitArray& resize(const size_t& bits)
    requires ResizableContainerConcept<Container, PackType>
    {
        this->_bits = bits;
        size_t req_size = _get_pack_size(bits);
        this->_data.resize(req_size);

        return *this;
    }

    /**
     * @brief BitArrayのビット数を取得します。
     * @return ビット数
     */
    size_t size() const {
        return this->_bits;
    }

    /**
     * @brief BitArrayの最大ビット数を取得します。
     * @return 最大ビット数
     */
    size_t max_size() const {
        return _get_bits_size(this->_data.max_size());
    }

    /**
     * @brief BitArrayのパック数を取得します。
     * @return パック数
     */
    size_t pack_size() const {
        return this->_data.size();
    }
    
    /**
     * @brief 指定された位置のビットを取得します。
     * @param n ビットインデックス
     * @return ビットの値
     */
    bool at(const size_t& n) const {
        return this->_get_bit(n);
    }

    /**
     * @brief 指定された位置のパックの値を取得します。
     * @param n パックインデックス
     * @return パックの値
     */
    PackType pack_at(const size_t& n) const {
        return this->_data.at(n);
    }

    /**
     * @brief 指定された位置のビットを設定します。
     * @param n ビットインデックス
     * @param val 設定する値
     */
    BitArray& set(const size_t& n, const bool& val){
        this->_set_bit(n, val);

        return *this;
    }

    /**
     * @brief BitArrayの各パックに値を設定します。
     * @param func 値を生成する関数
     */
    BitArray& set_pack(const size_t& n, const PackType& val){
        this->_data.at(n) = val;

        return *this;
    }

    /**
     * @brief BitArrayの各ビットに値を設定します。
     * @param func 値を生成する関数
     */
    template<typename Func> requires std::invocable<Func, size_t>
    BitArray& set_all(const Func& func){
        for(size_t i = 0; i < this->_bits; ++i){
            this->_set_bit(i, func(i));
        }

        return *this;
    }

    /**
     * @brief BitArrayの各パックに値を設定します。
     * @param func 値を生成する関数
     */
    template<typename Func> requires std::invocable<Func, size_t>
    BitArray& set_all_pack(const Func& func){
        for(size_t i = 0; i < this->pack_size(); ++i){
            this->_data.at(i) = func(i);
        }

        return *this;
    }

    /**
     * @brief ランダムな値でBitArrayを初期化します。
     * @param seed 乱数生成器のシード
     */
    BitArray& set_random(uint32_t seed = std::random_device{}()){
        std::default_random_engine engine(seed);
        std::uniform_int_distribution<uint32_t> dist(0, static_cast<uint32_t>(-1));

        for(size_t i = 0; i < this->pack_size(); ++i){
            this->_data.at(i) = static_cast<PackType>(dist(engine));
        }

        return *this;
    }

    /**
     * @brief BitArrayの末尾にビットを追加します。
     * @param val 追加するビットの値
     * @return 自身の参照
     */
    BitArray& emplace_back(const bool& val){
        size_t new_bits = this->_bits + 1;
        size_t req_size = _get_pack_size(new_bits);

        if(this->_data.max_size() < req_size)
            throw std::runtime_error("Container max_size is too small for the specified number of bits."); 

        if constexpr (ResizableContainerConcept<Container, PackType>) {
            if(req_size > this->_data.size())
                this->_data.emplace_back(static_cast<PackType>(0));
        } else {
            if(req_size > this->_data.size())
                throw std::runtime_error("Container size is too small for the specified number of bits.");
        }

        this->_bits = new_bits;
        this->_set_bit(this->_bits-1, val);

        return *this;
    }

    /**
     * @brief BitArrayを出力ストリームに書き込みます。
     * @param os 出力ストリーム
     * @param ba 書き込むBitArray
     * @return 出力ストリーム
     */
    friend std::ostream& operator << (std::ostream& os, const BitArray& ba){
        size_t total_bits = ba.size();
        for(size_t i = 0; i < ba.pack_size(); i++){
            if(total_bits < (i+1)*_PACKBITS){
                std::string b = std::bitset<sizeof(PackType) * 8>(ba.pack_at(i)).to_string();
                size_t remain = (i+1)*_PACKBITS - total_bits;
                
                os << b.substr(0, _PACKBITS - remain);
            }else{
                os << std::bitset<sizeof(PackType) * 8>(ba.pack_at(i)) << " ";
            }
        }
        return os;
    }
};
