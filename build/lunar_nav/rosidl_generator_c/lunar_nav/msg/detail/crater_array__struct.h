// generated from rosidl_generator_c/resource/idl__struct.h.em
// with input from lunar_nav:msg/CraterArray.idl
// generated code does not contain a copyright notice

#ifndef LUNAR_NAV__MSG__DETAIL__CRATER_ARRAY__STRUCT_H_
#define LUNAR_NAV__MSG__DETAIL__CRATER_ARRAY__STRUCT_H_

#ifdef __cplusplus
extern "C"
{
#endif

#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>


// Constants defined in the message

// Include directives for member types
// Member 'header'
#include "std_msgs/msg/detail/header__struct.h"
// Member 'craters'
#include "lunar_nav/msg/detail/crater__struct.h"

/// Struct defined in msg/CraterArray in the package lunar_nav.
typedef struct lunar_nav__msg__CraterArray
{
  std_msgs__msg__Header header;
  lunar_nav__msg__Crater__Sequence craters;
} lunar_nav__msg__CraterArray;

// Struct for a sequence of lunar_nav__msg__CraterArray.
typedef struct lunar_nav__msg__CraterArray__Sequence
{
  lunar_nav__msg__CraterArray * data;
  /// The number of valid items in data
  size_t size;
  /// The number of allocated items in data
  size_t capacity;
} lunar_nav__msg__CraterArray__Sequence;

#ifdef __cplusplus
}
#endif

#endif  // LUNAR_NAV__MSG__DETAIL__CRATER_ARRAY__STRUCT_H_
