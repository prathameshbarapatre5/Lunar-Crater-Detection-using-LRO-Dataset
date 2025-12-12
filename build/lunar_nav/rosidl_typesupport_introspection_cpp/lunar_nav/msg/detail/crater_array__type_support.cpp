// generated from rosidl_typesupport_introspection_cpp/resource/idl__type_support.cpp.em
// with input from lunar_nav:msg/CraterArray.idl
// generated code does not contain a copyright notice

#include "array"
#include "cstddef"
#include "string"
#include "vector"
#include "rosidl_runtime_c/message_type_support_struct.h"
#include "rosidl_typesupport_cpp/message_type_support.hpp"
#include "rosidl_typesupport_interface/macros.h"
#include "lunar_nav/msg/detail/crater_array__struct.hpp"
#include "rosidl_typesupport_introspection_cpp/field_types.hpp"
#include "rosidl_typesupport_introspection_cpp/identifier.hpp"
#include "rosidl_typesupport_introspection_cpp/message_introspection.hpp"
#include "rosidl_typesupport_introspection_cpp/message_type_support_decl.hpp"
#include "rosidl_typesupport_introspection_cpp/visibility_control.h"

namespace lunar_nav
{

namespace msg
{

namespace rosidl_typesupport_introspection_cpp
{

void CraterArray_init_function(
  void * message_memory, rosidl_runtime_cpp::MessageInitialization _init)
{
  new (message_memory) lunar_nav::msg::CraterArray(_init);
}

void CraterArray_fini_function(void * message_memory)
{
  auto typed_message = static_cast<lunar_nav::msg::CraterArray *>(message_memory);
  typed_message->~CraterArray();
}

size_t size_function__CraterArray__craters(const void * untyped_member)
{
  const auto * member = reinterpret_cast<const std::vector<lunar_nav::msg::Crater> *>(untyped_member);
  return member->size();
}

const void * get_const_function__CraterArray__craters(const void * untyped_member, size_t index)
{
  const auto & member =
    *reinterpret_cast<const std::vector<lunar_nav::msg::Crater> *>(untyped_member);
  return &member[index];
}

void * get_function__CraterArray__craters(void * untyped_member, size_t index)
{
  auto & member =
    *reinterpret_cast<std::vector<lunar_nav::msg::Crater> *>(untyped_member);
  return &member[index];
}

void fetch_function__CraterArray__craters(
  const void * untyped_member, size_t index, void * untyped_value)
{
  const auto & item = *reinterpret_cast<const lunar_nav::msg::Crater *>(
    get_const_function__CraterArray__craters(untyped_member, index));
  auto & value = *reinterpret_cast<lunar_nav::msg::Crater *>(untyped_value);
  value = item;
}

void assign_function__CraterArray__craters(
  void * untyped_member, size_t index, const void * untyped_value)
{
  auto & item = *reinterpret_cast<lunar_nav::msg::Crater *>(
    get_function__CraterArray__craters(untyped_member, index));
  const auto & value = *reinterpret_cast<const lunar_nav::msg::Crater *>(untyped_value);
  item = value;
}

void resize_function__CraterArray__craters(void * untyped_member, size_t size)
{
  auto * member =
    reinterpret_cast<std::vector<lunar_nav::msg::Crater> *>(untyped_member);
  member->resize(size);
}

static const ::rosidl_typesupport_introspection_cpp::MessageMember CraterArray_message_member_array[2] = {
  {
    "header",  // name
    ::rosidl_typesupport_introspection_cpp::ROS_TYPE_MESSAGE,  // type
    0,  // upper bound of string
    ::rosidl_typesupport_introspection_cpp::get_message_type_support_handle<std_msgs::msg::Header>(),  // members of sub message
    false,  // is array
    0,  // array size
    false,  // is upper bound
    offsetof(lunar_nav::msg::CraterArray, header),  // bytes offset in struct
    nullptr,  // default value
    nullptr,  // size() function pointer
    nullptr,  // get_const(index) function pointer
    nullptr,  // get(index) function pointer
    nullptr,  // fetch(index, &value) function pointer
    nullptr,  // assign(index, value) function pointer
    nullptr  // resize(index) function pointer
  },
  {
    "craters",  // name
    ::rosidl_typesupport_introspection_cpp::ROS_TYPE_MESSAGE,  // type
    0,  // upper bound of string
    ::rosidl_typesupport_introspection_cpp::get_message_type_support_handle<lunar_nav::msg::Crater>(),  // members of sub message
    true,  // is array
    0,  // array size
    false,  // is upper bound
    offsetof(lunar_nav::msg::CraterArray, craters),  // bytes offset in struct
    nullptr,  // default value
    size_function__CraterArray__craters,  // size() function pointer
    get_const_function__CraterArray__craters,  // get_const(index) function pointer
    get_function__CraterArray__craters,  // get(index) function pointer
    fetch_function__CraterArray__craters,  // fetch(index, &value) function pointer
    assign_function__CraterArray__craters,  // assign(index, value) function pointer
    resize_function__CraterArray__craters  // resize(index) function pointer
  }
};

static const ::rosidl_typesupport_introspection_cpp::MessageMembers CraterArray_message_members = {
  "lunar_nav::msg",  // message namespace
  "CraterArray",  // message name
  2,  // number of fields
  sizeof(lunar_nav::msg::CraterArray),
  CraterArray_message_member_array,  // message members
  CraterArray_init_function,  // function to initialize message memory (memory has to be allocated)
  CraterArray_fini_function  // function to terminate message instance (will not free memory)
};

static const rosidl_message_type_support_t CraterArray_message_type_support_handle = {
  ::rosidl_typesupport_introspection_cpp::typesupport_identifier,
  &CraterArray_message_members,
  get_message_typesupport_handle_function,
};

}  // namespace rosidl_typesupport_introspection_cpp

}  // namespace msg

}  // namespace lunar_nav


namespace rosidl_typesupport_introspection_cpp
{

template<>
ROSIDL_TYPESUPPORT_INTROSPECTION_CPP_PUBLIC
const rosidl_message_type_support_t *
get_message_type_support_handle<lunar_nav::msg::CraterArray>()
{
  return &::lunar_nav::msg::rosidl_typesupport_introspection_cpp::CraterArray_message_type_support_handle;
}

}  // namespace rosidl_typesupport_introspection_cpp

#ifdef __cplusplus
extern "C"
{
#endif

ROSIDL_TYPESUPPORT_INTROSPECTION_CPP_PUBLIC
const rosidl_message_type_support_t *
ROSIDL_TYPESUPPORT_INTERFACE__MESSAGE_SYMBOL_NAME(rosidl_typesupport_introspection_cpp, lunar_nav, msg, CraterArray)() {
  return &::lunar_nav::msg::rosidl_typesupport_introspection_cpp::CraterArray_message_type_support_handle;
}

#ifdef __cplusplus
}
#endif
