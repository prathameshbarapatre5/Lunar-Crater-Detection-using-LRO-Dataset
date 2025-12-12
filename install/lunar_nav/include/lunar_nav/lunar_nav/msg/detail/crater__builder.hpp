// generated from rosidl_generator_cpp/resource/idl__builder.hpp.em
// with input from lunar_nav:msg/Crater.idl
// generated code does not contain a copyright notice

#ifndef LUNAR_NAV__MSG__DETAIL__CRATER__BUILDER_HPP_
#define LUNAR_NAV__MSG__DETAIL__CRATER__BUILDER_HPP_

#include <algorithm>
#include <utility>

#include "lunar_nav/msg/detail/crater__struct.hpp"
#include "rosidl_runtime_cpp/message_initialization.hpp"


namespace lunar_nav
{

namespace msg
{

namespace builder
{

class Init_Crater_id
{
public:
  explicit Init_Crater_id(::lunar_nav::msg::Crater & msg)
  : msg_(msg)
  {}
  ::lunar_nav::msg::Crater id(::lunar_nav::msg::Crater::_id_type arg)
  {
    msg_.id = std::move(arg);
    return std::move(msg_);
  }

private:
  ::lunar_nav::msg::Crater msg_;
};

class Init_Crater_confidence
{
public:
  explicit Init_Crater_confidence(::lunar_nav::msg::Crater & msg)
  : msg_(msg)
  {}
  Init_Crater_id confidence(::lunar_nav::msg::Crater::_confidence_type arg)
  {
    msg_.confidence = std::move(arg);
    return Init_Crater_id(msg_);
  }

private:
  ::lunar_nav::msg::Crater msg_;
};

class Init_Crater_depth
{
public:
  explicit Init_Crater_depth(::lunar_nav::msg::Crater & msg)
  : msg_(msg)
  {}
  Init_Crater_confidence depth(::lunar_nav::msg::Crater::_depth_type arg)
  {
    msg_.depth = std::move(arg);
    return Init_Crater_confidence(msg_);
  }

private:
  ::lunar_nav::msg::Crater msg_;
};

class Init_Crater_radius
{
public:
  explicit Init_Crater_radius(::lunar_nav::msg::Crater & msg)
  : msg_(msg)
  {}
  Init_Crater_depth radius(::lunar_nav::msg::Crater::_radius_type arg)
  {
    msg_.radius = std::move(arg);
    return Init_Crater_depth(msg_);
  }

private:
  ::lunar_nav::msg::Crater msg_;
};

class Init_Crater_z
{
public:
  explicit Init_Crater_z(::lunar_nav::msg::Crater & msg)
  : msg_(msg)
  {}
  Init_Crater_radius z(::lunar_nav::msg::Crater::_z_type arg)
  {
    msg_.z = std::move(arg);
    return Init_Crater_radius(msg_);
  }

private:
  ::lunar_nav::msg::Crater msg_;
};

class Init_Crater_y
{
public:
  explicit Init_Crater_y(::lunar_nav::msg::Crater & msg)
  : msg_(msg)
  {}
  Init_Crater_z y(::lunar_nav::msg::Crater::_y_type arg)
  {
    msg_.y = std::move(arg);
    return Init_Crater_z(msg_);
  }

private:
  ::lunar_nav::msg::Crater msg_;
};

class Init_Crater_x
{
public:
  Init_Crater_x()
  : msg_(::rosidl_runtime_cpp::MessageInitialization::SKIP)
  {}
  Init_Crater_y x(::lunar_nav::msg::Crater::_x_type arg)
  {
    msg_.x = std::move(arg);
    return Init_Crater_y(msg_);
  }

private:
  ::lunar_nav::msg::Crater msg_;
};

}  // namespace builder

}  // namespace msg

template<typename MessageType>
auto build();

template<>
inline
auto build<::lunar_nav::msg::Crater>()
{
  return lunar_nav::msg::builder::Init_Crater_x();
}

}  // namespace lunar_nav

#endif  // LUNAR_NAV__MSG__DETAIL__CRATER__BUILDER_HPP_
