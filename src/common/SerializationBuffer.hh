#ifndef CHIP_SERIALIZATION_BUFFER_HH
#define CHIP_SERIALIZATION_BUFFER_HH

#include <vector>
#include <string>
#include <cstring>
#include <stdexcept>

namespace chipstar {

class SerializationBuffer {
private:
  std::vector<char> Buffer_;
  size_t ReadPos_ = 0;

public:
  // Writing methods
  template<typename T>
  void write(const T& Value) {
    write(&Value, sizeof(T));
  }

  void write(const std::string& Value) {
    size_t Length = Value.length();
    write(Length);
    write(Value.data(), Length);
  }

  void write(const void* Data, size_t Size) {
    size_t OldSize = Buffer_.size();
    Buffer_.resize(OldSize + Size);
    std::memcpy(Buffer_.data() + OldSize, Data, Size);
  }

  // Reading methods
  template<typename T>
  void read(T& Value) {
    read(&Value, sizeof(T));
  }

  template<typename T>
  T read() {
    T Value;
    read(&Value, sizeof(T));
    return Value;
  }

  void read(std::string& Value) {
    size_t Length;
    read(Length);
    
    if (ReadPos_ + Length > Buffer_.size())
      throw std::runtime_error("Buffer underflow during string deserialization");
      
    Value.assign(Buffer_.data() + ReadPos_, Length);
    ReadPos_ += Length;
  }

  void read(void* Data, size_t Size) {
    if (ReadPos_ + Size > Buffer_.size())
      throw std::runtime_error("Buffer underflow during deserialization");
      
    std::memcpy(Data, Buffer_.data() + ReadPos_, Size);
    ReadPos_ += Size;
  }

  // Utility methods
  void clear() {
    Buffer_.clear();
    ReadPos_ = 0;
  }

  size_t size() const {
    return Buffer_.size();
  }

  const char* data() const {
    return Buffer_.data();
  }

  void setBuffer(const void* Data, size_t Size) {
    Buffer_.resize(Size);
    std::memcpy(Buffer_.data(), Data, Size);
    ReadPos_ = 0;
  }

  bool hasMore() const {
    return ReadPos_ < Buffer_.size();
  }
};

} // namespace chipstar

#endif // CHIP_SERIALIZATION_BUFFER_HH 