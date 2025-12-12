// generated from rosidl_generator_cpp/resource/idl__traits.hpp.em
// with input from lunar_nav:msg/Crater.idl
// generated code does not contain a copyright notice

#ifndef LUNAR_NAV__MSG__DETAIL__CRATER__TRAITS_HPP_
#define LUNAR_NAV__MSG__DETAIL__CRATER__TRAITS_HPP_

#include <stdint.h>

#include <sstream>
#include <string>
#include <type_traits>

#include "lunar_nav/msg/detail/crater__struct.hpp"
#include "rosidl_runtime_cpp/traits.hpp"

namespace lunar_nav
{

namespace msg
{

inline void to_flow_style_yaml(
  const Crater & msg,
  std::ostream & out)
{
  out << "{";
  // member: x
  {
    out << "x: ";
    rosidl_generator_traits::value_to_yaml(msg.x, out);
    out << ", ";
  }

  // member: y
  {
    out << "y: ";
    rosidl_generator_traits::value_to_yaml(msg.y, out);
    out << ", ";
  }

  // member: z
  {
    out << "z: ";
    rosidl_generator_traits::value_to_yaml(msg.z, out);
    out << ", ";
  }

  // member: radius
  {
    out << "radius: ";
    rosidl_generator_traits::value_to_yaml(msg.radius, out);
    out << ", ";
  }

  // member: depth
  {
    out << "depth: ";
    rosidl_generator_traits::value_to_yaml(msg.depth, out);
    out << ", ";
  }

  // member: confidence
  {
    out << "confidence: ";
    rosidl_generator_traits::value_to_yaml(msg.confidence, out);
    out << ", ";
  }

  // member: id
  {
    out << "id: ";
    rosidl_generator_traits::value_to_yaml(msg.id, out);
  }
  out << "}";
}  // NOLINT(readability/fn_size)

inline void to_block_style_yaml(
  const Crater & msg,
  std::ostream & out, size_t indentation = 0)
{
  // member: x
  {
    if (indentation > 0) {
      out << std::string(indentation, ' ');
    }
    out << "x: ";
    rosidl_generator_traits::value_to_yaml(msg.x, out);
    out << "\n";
  }

  // member: y
  {
    if (indentation > 0) {
      out << std::string(indentation, ' ');
    }
    out << "y: ";
    rosidl_generator_traits::value_to_yaml(msg.y, out);
    out << "\n";
  }

  // member: z
  {
    if (indentation > 0) {
      out << std::string(indentation, ' ');
    }
    out << "z: ";
    rosidl_generator_traits::value_to_yaml(msg.z, out);
    out << "\n";
  }

  // member: radius
  {
    if (indentation > 0) {
      out << std::string(indentation, ' ');
    }
    out << "radius: ";
    rosidl_generator_traits::value_to_yaml(msg.radius, out);
    out << "\n";
  }

  // member: depth
  {
    if (indentation > 0) {
      out << std::string(indentation, ' ');
    }
    out << "depth: ";
    rosidl_generator_traits::value_to_yaml(msg.depth, out);
    out << "\n";
  }

  // member: confidence
  {
    if (indentation > 0) {
      out << std::string(indentation, ' ');
    }
    out << "confidence: ";
    rosidl_generator_traits::value_to_yaml(msg.confidence, out);
    out << "\n";
  }

  // member: id
  {
    if (indentation > 0) {
      out << std::string(indentation, ' ');
    }
    out << "id: ";
    rosidl_generator_traits::value_to_yaml(msg.id, out);
    out << "\n";
  }
}  // NOLINT(readability/fn_size)

inline std::string to_yaml(const Crater & msg, bool use_flow_style = false)
{
  std::ostringstream out;
  if (use_flow_style) {
    to_flow_style_yaml(msg, out);
  } else {
    to_block_style_yaml(msg, out);
  }
  return out.str();
}

}  // namespace msg

}  // namespace lunar_nav

namespace rosidl_generator_traits
{

[[deprecated("use lunar_nav::msg::to_block_style_yaml() instead")]]
inline void to_yaml(
  const lunar_nav::msg::Crater & msg,
  std::ostream & out, size_t indentation = 0)
{
  lunar_nav::msg::to_block_style_yaml(msg, out, indentation);
}

[[deprecated("use lunar_nav::msg::to_yaml() instead")]]
inline std::string to_yaml(const lunar_nav::msg::Crater & msg)
{
  return lunar_nav::msg::to_yaml(msg);
}

template<>
inline const char * data_type<lunar_nav::msg::Crater>()
{
  return "lunar_nav::msg::Crater";
}

template<>
inline const char * name<lunar_nav::msg::Crater>()
{
  return "lunar_nav/msg/Crater";
}

template<>
struct has_fixed_size<lunar_nav::msg::Crater>
  : std::integral_constant<bool, true> {};

template<>
struct has_bounded_size<lunar_nav::msg::Crater>
  : std::integral_constant<bool, true> {};

template<>
struct is_message<lunar_nav::msg::Crater>
  : std::true_type {};

}  // namespace rosidl_generator_traits

#endif  // LUNAR_NAV__MSG__DETAIL__CRATER__TRAITS_HPP_
