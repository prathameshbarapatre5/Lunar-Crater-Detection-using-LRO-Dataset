// generated from rosidl_generator_cpp/resource/idl__struct.hpp.em
// with input from lunar_nav:msg/CraterArray.idl
// generated code does not contain a copyright notice

#ifndef LUNAR_NAV__MSG__DETAIL__CRATER_ARRAY__STRUCT_HPP_
#define LUNAR_NAV__MSG__DETAIL__CRATER_ARRAY__STRUCT_HPP_

#include <algorithm>
#include <array>
#include <memory>
#include <string>
#include <vector>

#include "rosidl_runtime_cpp/bounded_vector.hpp"
#include "rosidl_runtime_cpp/message_initialization.hpp"


// Include directives for member types
// Member 'header'
#include "std_msgs/msg/detail/header__struct.hpp"
// Member 'craters'
#include "lunar_nav/msg/detail/crater__struct.hpp"

#ifndef _WIN32
# define DEPRECATED__lunar_nav__msg__CraterArray __attribute__((deprecated))
#else
# define DEPRECATED__lunar_nav__msg__CraterArray __declspec(deprecated)
#endif

namespace lunar_nav
{

namespace msg
{

// message struct
template<class ContainerAllocator>
struct CraterArray_
{
  using Type = CraterArray_<ContainerAllocator>;

  explicit CraterArray_(rosidl_runtime_cpp::MessageInitialization _init = rosidl_runtime_cpp::MessageInitialization::ALL)
  : header(_init)
  {
    (void)_init;
  }

  explicit CraterArray_(const ContainerAllocator & _alloc, rosidl_runtime_cpp::MessageInitialization _init = rosidl_runtime_cpp::MessageInitialization::ALL)
  : header(_alloc, _init)
  {
    (void)_init;
  }

  // field types and members
  using _header_type =
    std_msgs::msg::Header_<ContainerAllocator>;
  _header_type header;
  using _craters_type =
    std::vector<lunar_nav::msg::Crater_<ContainerAllocator>, typename std::allocator_traits<ContainerAllocator>::template rebind_alloc<lunar_nav::msg::Crater_<ContainerAllocator>>>;
  _craters_type craters;

  // setters for named parameter idiom
  Type & set__header(
    const std_msgs::msg::Header_<ContainerAllocator> & _arg)
  {
    this->header = _arg;
    return *this;
  }
  Type & set__craters(
    const std::vector<lunar_nav::msg::Crater_<ContainerAllocator>, typename std::allocator_traits<ContainerAllocator>::template rebind_alloc<lunar_nav::msg::Crater_<ContainerAllocator>>> & _arg)
  {
    this->craters = _arg;
    return *this;
  }

  // constant declarations

  // pointer types
  using RawPtr =
    lunar_nav::msg::CraterArray_<ContainerAllocator> *;
  using ConstRawPtr =
    const lunar_nav::msg::CraterArray_<ContainerAllocator> *;
  using SharedPtr =
    std::shared_ptr<lunar_nav::msg::CraterArray_<ContainerAllocator>>;
  using ConstSharedPtr =
    std::shared_ptr<lunar_nav::msg::CraterArray_<ContainerAllocator> const>;

  template<typename Deleter = std::default_delete<
      lunar_nav::msg::CraterArray_<ContainerAllocator>>>
  using UniquePtrWithDeleter =
    std::unique_ptr<lunar_nav::msg::CraterArray_<ContainerAllocator>, Deleter>;

  using UniquePtr = UniquePtrWithDeleter<>;

  template<typename Deleter = std::default_delete<
      lunar_nav::msg::CraterArray_<ContainerAllocator>>>
  using ConstUniquePtrWithDeleter =
    std::unique_ptr<lunar_nav::msg::CraterArray_<ContainerAllocator> const, Deleter>;
  using ConstUniquePtr = ConstUniquePtrWithDeleter<>;

  using WeakPtr =
    std::weak_ptr<lunar_nav::msg::CraterArray_<ContainerAllocator>>;
  using ConstWeakPtr =
    std::weak_ptr<lunar_nav::msg::CraterArray_<ContainerAllocator> const>;

  // pointer types similar to ROS 1, use SharedPtr / ConstSharedPtr instead
  // NOTE: Can't use 'using' here because GNU C++ can't parse attributes properly
  typedef DEPRECATED__lunar_nav__msg__CraterArray
    std::shared_ptr<lunar_nav::msg::CraterArray_<ContainerAllocator>>
    Ptr;
  typedef DEPRECATED__lunar_nav__msg__CraterArray
    std::shared_ptr<lunar_nav::msg::CraterArray_<ContainerAllocator> const>
    ConstPtr;

  // comparison operators
  bool operator==(const CraterArray_ & other) const
  {
    if (this->header != other.header) {
      return false;
    }
    if (this->craters != other.craters) {
      return false;
    }
    return true;
  }
  bool operator!=(const CraterArray_ & other) const
  {
    return !this->operator==(other);
  }
};  // struct CraterArray_

// alias to use template instance with default allocator
using CraterArray =
  lunar_nav::msg::CraterArray_<std::allocator<void>>;

// constant definitions

}  // namespace msg

}  // namespace lunar_nav

#endif  // LUNAR_NAV__MSG__DETAIL__CRATER_ARRAY__STRUCT_HPP_
