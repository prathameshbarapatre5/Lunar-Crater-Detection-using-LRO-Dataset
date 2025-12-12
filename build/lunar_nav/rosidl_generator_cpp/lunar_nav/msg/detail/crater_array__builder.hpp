// generated from rosidl_generator_cpp/resource/idl__builder.hpp.em
// with input from lunar_nav:msg/CraterArray.idl
// generated code does not contain a copyright notice

#ifndef LUNAR_NAV__MSG__DETAIL__CRATER_ARRAY__BUILDER_HPP_
#define LUNAR_NAV__MSG__DETAIL__CRATER_ARRAY__BUILDER_HPP_

#include <algorithm>
#include <utility>

#include "lunar_nav/msg/detail/crater_array__struct.hpp"
#include "rosidl_runtime_cpp/message_initialization.hpp"


namespace lunar_nav
{

namespace msg
{

namespace builder
{

class Init_CraterArray_craters
{
public:
  explicit Init_CraterArray_craters(::lunar_nav::msg::CraterArray & msg)
  : msg_(msg)
  {}
  ::lunar_nav::msg::CraterArray craters(::lunar_nav::msg::CraterArray::_craters_type arg)
  {
    msg_.craters = std::move(arg);
    return std::move(msg_);
  }

private:
  ::lunar_nav::msg::CraterArray msg_;
};

class Init_CraterArray_header
{
public:
  Init_CraterArray_header()
  : msg_(::rosidl_runtime_cpp::MessageInitialization::SKIP)
  {}
  Init_CraterArray_craters header(::lunar_nav::msg::CraterArray::_header_type arg)
  {
    msg_.header = std::move(arg);
    return Init_CraterArray_craters(msg_);
  }

private:
  ::lunar_nav::msg::CraterArray msg_;
};

}  // namespace builder

}  // namespace msg

template<typename MessageType>
auto build();

template<>
inline
auto build<::lunar_nav::msg::CraterArray>()
{
  return lunar_nav::msg::builder::Init_CraterArray_header();
}

}  // namespace lunar_nav

#endif  // LUNAR_NAV__MSG__DETAIL__CRATER_ARRAY__BUILDER_HPP_
