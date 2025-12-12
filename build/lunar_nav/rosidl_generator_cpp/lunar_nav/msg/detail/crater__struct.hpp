// generated from rosidl_generator_cpp/resource/idl__struct.hpp.em
// with input from lunar_nav:msg/Crater.idl
// generated code does not contain a copyright notice

#ifndef LUNAR_NAV__MSG__DETAIL__CRATER__STRUCT_HPP_
#define LUNAR_NAV__MSG__DETAIL__CRATER__STRUCT_HPP_

#include <algorithm>
#include <array>
#include <memory>
#include <string>
#include <vector>

#include "rosidl_runtime_cpp/bounded_vector.hpp"
#include "rosidl_runtime_cpp/message_initialization.hpp"


#ifndef _WIN32
# define DEPRECATED__lunar_nav__msg__Crater __attribute__((deprecated))
#else
# define DEPRECATED__lunar_nav__msg__Crater __declspec(deprecated)
#endif

namespace lunar_nav
{

namespace msg
{

// message struct
template<class ContainerAllocator>
struct Crater_
{
  using Type = Crater_<ContainerAllocator>;

  explicit Crater_(rosidl_runtime_cpp::MessageInitialization _init = rosidl_runtime_cpp::MessageInitialization::ALL)
  {
    if (rosidl_runtime_cpp::MessageInitialization::ALL == _init ||
      rosidl_runtime_cpp::MessageInitialization::ZERO == _init)
    {
      this->x = 0.0;
      this->y = 0.0;
      this->z = 0.0;
      this->radius = 0.0;
      this->depth = 0.0;
      this->confidence = 0.0;
      this->id = 0l;
    }
  }

  explicit Crater_(const ContainerAllocator & _alloc, rosidl_runtime_cpp::MessageInitialization _init = rosidl_runtime_cpp::MessageInitialization::ALL)
  {
    (void)_alloc;
    if (rosidl_runtime_cpp::MessageInitialization::ALL == _init ||
      rosidl_runtime_cpp::MessageInitialization::ZERO == _init)
    {
      this->x = 0.0;
      this->y = 0.0;
      this->z = 0.0;
      this->radius = 0.0;
      this->depth = 0.0;
      this->confidence = 0.0;
      this->id = 0l;
    }
  }

  // field types and members
  using _x_type =
    double;
  _x_type x;
  using _y_type =
    double;
  _y_type y;
  using _z_type =
    double;
  _z_type z;
  using _radius_type =
    double;
  _radius_type radius;
  using _depth_type =
    double;
  _depth_type depth;
  using _confidence_type =
    double;
  _confidence_type confidence;
  using _id_type =
    int32_t;
  _id_type id;

  // setters for named parameter idiom
  Type & set__x(
    const double & _arg)
  {
    this->x = _arg;
    return *this;
  }
  Type & set__y(
    const double & _arg)
  {
    this->y = _arg;
    return *this;
  }
  Type & set__z(
    const double & _arg)
  {
    this->z = _arg;
    return *this;
  }
  Type & set__radius(
    const double & _arg)
  {
    this->radius = _arg;
    return *this;
  }
  Type & set__depth(
    const double & _arg)
  {
    this->depth = _arg;
    return *this;
  }
  Type & set__confidence(
    const double & _arg)
  {
    this->confidence = _arg;
    return *this;
  }
  Type & set__id(
    const int32_t & _arg)
  {
    this->id = _arg;
    return *this;
  }

  // constant declarations

  // pointer types
  using RawPtr =
    lunar_nav::msg::Crater_<ContainerAllocator> *;
  using ConstRawPtr =
    const lunar_nav::msg::Crater_<ContainerAllocator> *;
  using SharedPtr =
    std::shared_ptr<lunar_nav::msg::Crater_<ContainerAllocator>>;
  using ConstSharedPtr =
    std::shared_ptr<lunar_nav::msg::Crater_<ContainerAllocator> const>;

  template<typename Deleter = std::default_delete<
      lunar_nav::msg::Crater_<ContainerAllocator>>>
  using UniquePtrWithDeleter =
    std::unique_ptr<lunar_nav::msg::Crater_<ContainerAllocator>, Deleter>;

  using UniquePtr = UniquePtrWithDeleter<>;

  template<typename Deleter = std::default_delete<
      lunar_nav::msg::Crater_<ContainerAllocator>>>
  using ConstUniquePtrWithDeleter =
    std::unique_ptr<lunar_nav::msg::Crater_<ContainerAllocator> const, Deleter>;
  using ConstUniquePtr = ConstUniquePtrWithDeleter<>;

  using WeakPtr =
    std::weak_ptr<lunar_nav::msg::Crater_<ContainerAllocator>>;
  using ConstWeakPtr =
    std::weak_ptr<lunar_nav::msg::Crater_<ContainerAllocator> const>;

  // pointer types similar to ROS 1, use SharedPtr / ConstSharedPtr instead
  // NOTE: Can't use 'using' here because GNU C++ can't parse attributes properly
  typedef DEPRECATED__lunar_nav__msg__Crater
    std::shared_ptr<lunar_nav::msg::Crater_<ContainerAllocator>>
    Ptr;
  typedef DEPRECATED__lunar_nav__msg__Crater
    std::shared_ptr<lunar_nav::msg::Crater_<ContainerAllocator> const>
    ConstPtr;

  // comparison operators
  bool operator==(const Crater_ & other) const
  {
    if (this->x != other.x) {
      return false;
    }
    if (this->y != other.y) {
      return false;
    }
    if (this->z != other.z) {
      return false;
    }
    if (this->radius != other.radius) {
      return false;
    }
    if (this->depth != other.depth) {
      return false;
    }
    if (this->confidence != other.confidence) {
      return false;
    }
    if (this->id != other.id) {
      return false;
    }
    return true;
  }
  bool operator!=(const Crater_ & other) const
  {
    return !this->operator==(other);
  }
};  // struct Crater_

// alias to use template instance with default allocator
using Crater =
  lunar_nav::msg::Crater_<std::allocator<void>>;

// constant definitions

}  // namespace msg

}  // namespace lunar_nav

#endif  // LUNAR_NAV__MSG__DETAIL__CRATER__STRUCT_HPP_
