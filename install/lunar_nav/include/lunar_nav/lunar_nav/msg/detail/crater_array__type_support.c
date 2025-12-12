// generated from rosidl_typesupport_introspection_c/resource/idl__type_support.c.em
// with input from lunar_nav:msg/CraterArray.idl
// generated code does not contain a copyright notice

#include <stddef.h>
#include "lunar_nav/msg/detail/crater_array__rosidl_typesupport_introspection_c.h"
#include "lunar_nav/msg/rosidl_typesupport_introspection_c__visibility_control.h"
#include "rosidl_typesupport_introspection_c/field_types.h"
#include "rosidl_typesupport_introspection_c/identifier.h"
#include "rosidl_typesupport_introspection_c/message_introspection.h"
#include "lunar_nav/msg/detail/crater_array__functions.h"
#include "lunar_nav/msg/detail/crater_array__struct.h"


// Include directives for member types
// Member `header`
#include "std_msgs/msg/header.h"
// Member `header`
#include "std_msgs/msg/detail/header__rosidl_typesupport_introspection_c.h"
// Member `craters`
#include "lunar_nav/msg/crater.h"
// Member `craters`
#include "lunar_nav/msg/detail/crater__rosidl_typesupport_introspection_c.h"

#ifdef __cplusplus
extern "C"
{
#endif

void lunar_nav__msg__CraterArray__rosidl_typesupport_introspection_c__CraterArray_init_function(
  void * message_memory, enum rosidl_runtime_c__message_initialization _init)
{
  // TODO(karsten1987): initializers are not yet implemented for typesupport c
  // see https://github.com/ros2/ros2/issues/397
  (void) _init;
  lunar_nav__msg__CraterArray__init(message_memory);
}

void lunar_nav__msg__CraterArray__rosidl_typesupport_introspection_c__CraterArray_fini_function(void * message_memory)
{
  lunar_nav__msg__CraterArray__fini(message_memory);
}

size_t lunar_nav__msg__CraterArray__rosidl_typesupport_introspection_c__size_function__CraterArray__craters(
  const void * untyped_member)
{
  const lunar_nav__msg__Crater__Sequence * member =
    (const lunar_nav__msg__Crater__Sequence *)(untyped_member);
  return member->size;
}

const void * lunar_nav__msg__CraterArray__rosidl_typesupport_introspection_c__get_const_function__CraterArray__craters(
  const void * untyped_member, size_t index)
{
  const lunar_nav__msg__Crater__Sequence * member =
    (const lunar_nav__msg__Crater__Sequence *)(untyped_member);
  return &member->data[index];
}

void * lunar_nav__msg__CraterArray__rosidl_typesupport_introspection_c__get_function__CraterArray__craters(
  void * untyped_member, size_t index)
{
  lunar_nav__msg__Crater__Sequence * member =
    (lunar_nav__msg__Crater__Sequence *)(untyped_member);
  return &member->data[index];
}

void lunar_nav__msg__CraterArray__rosidl_typesupport_introspection_c__fetch_function__CraterArray__craters(
  const void * untyped_member, size_t index, void * untyped_value)
{
  const lunar_nav__msg__Crater * item =
    ((const lunar_nav__msg__Crater *)
    lunar_nav__msg__CraterArray__rosidl_typesupport_introspection_c__get_const_function__CraterArray__craters(untyped_member, index));
  lunar_nav__msg__Crater * value =
    (lunar_nav__msg__Crater *)(untyped_value);
  *value = *item;
}

void lunar_nav__msg__CraterArray__rosidl_typesupport_introspection_c__assign_function__CraterArray__craters(
  void * untyped_member, size_t index, const void * untyped_value)
{
  lunar_nav__msg__Crater * item =
    ((lunar_nav__msg__Crater *)
    lunar_nav__msg__CraterArray__rosidl_typesupport_introspection_c__get_function__CraterArray__craters(untyped_member, index));
  const lunar_nav__msg__Crater * value =
    (const lunar_nav__msg__Crater *)(untyped_value);
  *item = *value;
}

bool lunar_nav__msg__CraterArray__rosidl_typesupport_introspection_c__resize_function__CraterArray__craters(
  void * untyped_member, size_t size)
{
  lunar_nav__msg__Crater__Sequence * member =
    (lunar_nav__msg__Crater__Sequence *)(untyped_member);
  lunar_nav__msg__Crater__Sequence__fini(member);
  return lunar_nav__msg__Crater__Sequence__init(member, size);
}

static rosidl_typesupport_introspection_c__MessageMember lunar_nav__msg__CraterArray__rosidl_typesupport_introspection_c__CraterArray_message_member_array[2] = {
  {
    "header",  // name
    rosidl_typesupport_introspection_c__ROS_TYPE_MESSAGE,  // type
    0,  // upper bound of string
    NULL,  // members of sub message (initialized later)
    false,  // is array
    0,  // array size
    false,  // is upper bound
    offsetof(lunar_nav__msg__CraterArray, header),  // bytes offset in struct
    NULL,  // default value
    NULL,  // size() function pointer
    NULL,  // get_const(index) function pointer
    NULL,  // get(index) function pointer
    NULL,  // fetch(index, &value) function pointer
    NULL,  // assign(index, value) function pointer
    NULL  // resize(index) function pointer
  },
  {
    "craters",  // name
    rosidl_typesupport_introspection_c__ROS_TYPE_MESSAGE,  // type
    0,  // upper bound of string
    NULL,  // members of sub message (initialized later)
    true,  // is array
    0,  // array size
    false,  // is upper bound
    offsetof(lunar_nav__msg__CraterArray, craters),  // bytes offset in struct
    NULL,  // default value
    lunar_nav__msg__CraterArray__rosidl_typesupport_introspection_c__size_function__CraterArray__craters,  // size() function pointer
    lunar_nav__msg__CraterArray__rosidl_typesupport_introspection_c__get_const_function__CraterArray__craters,  // get_const(index) function pointer
    lunar_nav__msg__CraterArray__rosidl_typesupport_introspection_c__get_function__CraterArray__craters,  // get(index) function pointer
    lunar_nav__msg__CraterArray__rosidl_typesupport_introspection_c__fetch_function__CraterArray__craters,  // fetch(index, &value) function pointer
    lunar_nav__msg__CraterArray__rosidl_typesupport_introspection_c__assign_function__CraterArray__craters,  // assign(index, value) function pointer
    lunar_nav__msg__CraterArray__rosidl_typesupport_introspection_c__resize_function__CraterArray__craters  // resize(index) function pointer
  }
};

static const rosidl_typesupport_introspection_c__MessageMembers lunar_nav__msg__CraterArray__rosidl_typesupport_introspection_c__CraterArray_message_members = {
  "lunar_nav__msg",  // message namespace
  "CraterArray",  // message name
  2,  // number of fields
  sizeof(lunar_nav__msg__CraterArray),
  lunar_nav__msg__CraterArray__rosidl_typesupport_introspection_c__CraterArray_message_member_array,  // message members
  lunar_nav__msg__CraterArray__rosidl_typesupport_introspection_c__CraterArray_init_function,  // function to initialize message memory (memory has to be allocated)
  lunar_nav__msg__CraterArray__rosidl_typesupport_introspection_c__CraterArray_fini_function  // function to terminate message instance (will not free memory)
};

// this is not const since it must be initialized on first access
// since C does not allow non-integral compile-time constants
static rosidl_message_type_support_t lunar_nav__msg__CraterArray__rosidl_typesupport_introspection_c__CraterArray_message_type_support_handle = {
  0,
  &lunar_nav__msg__CraterArray__rosidl_typesupport_introspection_c__CraterArray_message_members,
  get_message_typesupport_handle_function,
};

ROSIDL_TYPESUPPORT_INTROSPECTION_C_EXPORT_lunar_nav
const rosidl_message_type_support_t *
ROSIDL_TYPESUPPORT_INTERFACE__MESSAGE_SYMBOL_NAME(rosidl_typesupport_introspection_c, lunar_nav, msg, CraterArray)() {
  lunar_nav__msg__CraterArray__rosidl_typesupport_introspection_c__CraterArray_message_member_array[0].members_ =
    ROSIDL_TYPESUPPORT_INTERFACE__MESSAGE_SYMBOL_NAME(rosidl_typesupport_introspection_c, std_msgs, msg, Header)();
  lunar_nav__msg__CraterArray__rosidl_typesupport_introspection_c__CraterArray_message_member_array[1].members_ =
    ROSIDL_TYPESUPPORT_INTERFACE__MESSAGE_SYMBOL_NAME(rosidl_typesupport_introspection_c, lunar_nav, msg, Crater)();
  if (!lunar_nav__msg__CraterArray__rosidl_typesupport_introspection_c__CraterArray_message_type_support_handle.typesupport_identifier) {
    lunar_nav__msg__CraterArray__rosidl_typesupport_introspection_c__CraterArray_message_type_support_handle.typesupport_identifier =
      rosidl_typesupport_introspection_c__identifier;
  }
  return &lunar_nav__msg__CraterArray__rosidl_typesupport_introspection_c__CraterArray_message_type_support_handle;
}
#ifdef __cplusplus
}
#endif
