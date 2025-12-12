// generated from rosidl_generator_c/resource/idl__struct.h.em
// with input from lunar_nav:msg/Crater.idl
// generated code does not contain a copyright notice

#ifndef LUNAR_NAV__MSG__DETAIL__CRATER__STRUCT_H_
#define LUNAR_NAV__MSG__DETAIL__CRATER__STRUCT_H_

#ifdef __cplusplus
extern "C"
{
#endif

#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>


// Constants defined in the message

/// Struct defined in msg/Crater in the package lunar_nav.
typedef struct lunar_nav__msg__Crater
{
  double x;
  double y;
  double z;
  double radius;
  double depth;
  double confidence;
  int32_t id;
} lunar_nav__msg__Crater;

// Struct for a sequence of lunar_nav__msg__Crater.
typedef struct lunar_nav__msg__Crater__Sequence
{
  lunar_nav__msg__Crater * data;
  /// The number of valid items in data
  size_t size;
  /// The number of allocated items in data
  size_t capacity;
} lunar_nav__msg__Crater__Sequence;

#ifdef __cplusplus
}
#endif

#endif  // LUNAR_NAV__MSG__DETAIL__CRATER__STRUCT_H_
