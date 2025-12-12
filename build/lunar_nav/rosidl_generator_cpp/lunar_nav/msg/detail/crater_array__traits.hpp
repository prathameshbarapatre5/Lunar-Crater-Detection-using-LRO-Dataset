// generated from rosidl_generator_cpp/resource/idl__traits.hpp.em
// with input from lunar_nav:msg/CraterArray.idl
// generated code does not contain a copyright notice

#ifndef LUNAR_NAV__MSG__DETAIL__CRATER_ARRAY__TRAITS_HPP_
#define LUNAR_NAV__MSG__DETAIL__CRATER_ARRAY__TRAITS_HPP_

#include <stdint.h>

#include <sstream>
#include <string>
#include <type_traits>

#include "lunar_nav/msg/detail/crater_array__struct.hpp"
#include "rosidl_runtime_cpp/traits.hpp"

// Include directives for member types
// Member 'header'
#include "std_msgs/msg/detail/header__traits.hpp"
// Member 'craters'
#include "lunar_nav/msg/detail/crater__traits.hpp"

namespace lunar_nav
{

namespace msg
{

inline void to_flow_style_yaml(
  const CraterArray & msg,
  std::ostream & out)
{
  out << "{";
  // member: header
  {
    out << "header: ";
    to_flow_style_yaml(msg.header, out);
    out << ", ";
  }

  // member: craters
  {
    if (msg.craters.size() == 0) {
      out << "craters: []";
    } else {
      out << "craters: [";
      size_t pending_items = msg.craters.size();
      for (auto item : msg.craters) {
        to_flow_style_yaml(item, out);
        if (--pending_items > 0) {
          out << ", ";
        }
      }
      out << "]";
    }
  }
  out << "}";
}  // NOLINT(readability/fn_size)

inline void to_block_style_yaml(
  const CraterArray & msg,
  std::ostream & out, size_t indentation = 0)
{
  // member: header
  {
    if (indentation > 0) {
      out << std::string(indentation, ' ');
    }
    out << "header:\n";
    to_block_style_yaml(msg.header, out, indentation + 2);
  }

  // member: craters
  {
    if (indentation > 0) {
      out << std::string(indentation, ' ');
    }
    if (msg.craters.size() == 0) {
      out << "craters: []\n";
    } else {
      out << "craters:\n";
      for (auto item : msg.craters) {
        if (indentation > 0) {
          out << std::string(indentation, ' ');
        }
        out << "-\n";
        to_block_style_yaml(item, out, indentation + 2);
      }
    }
  }
}  // NOLINT(readability/fn_size)

inline std::string to_yaml(const CraterArray & msg, bool use_flow_style = false)
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
  const lunar_nav::msg::CraterArray & msg,
  std::ostream & out, size_t indentation = 0)
{
  lunar_nav::msg::to_block_style_yaml(msg, out, indentation);
}

[[deprecated("use lunar_nav::msg::to_yaml() instead")]]
inline std::string to_yaml(const lunar_nav::msg::CraterArray & msg)
{
  return lunar_nav::msg::to_yaml(msg);
}

template<>
inline const char * data_type<lunar_nav::msg::CraterArray>()
{
  return "lunar_nav::msg::CraterArray";
}

template<>
inline const char * name<lunar_nav::msg::CraterArray>()
{
  return "lunar_nav/msg/CraterArray";
}

template<>
struct has_fixed_size<lunar_nav::msg::CraterArray>
  : std::integral_constant<bool, false> {};

template<>
struct has_bounded_size<lunar_nav::msg::CraterArray>
  : std::integral_constant<bool, false> {};

template<>
struct is_message<lunar_nav::msg::CraterArray>
  : std::true_type {};

}  // namespace rosidl_generator_traits

#endif  // LUNAR_NAV__MSG__DETAIL__CRATER_ARRAY__TRAITS_HPP_
