// generated from rosidl_generator_c/resource/idl__functions.h.em
// with input from lunar_nav:msg/Crater.idl
// generated code does not contain a copyright notice

#ifndef LUNAR_NAV__MSG__DETAIL__CRATER__FUNCTIONS_H_
#define LUNAR_NAV__MSG__DETAIL__CRATER__FUNCTIONS_H_

#ifdef __cplusplus
extern "C"
{
#endif

#include <stdbool.h>
#include <stdlib.h>

#include "rosidl_runtime_c/visibility_control.h"
#include "lunar_nav/msg/rosidl_generator_c__visibility_control.h"

#include "lunar_nav/msg/detail/crater__struct.h"

/// Initialize msg/Crater message.
/**
 * If the init function is called twice for the same message without
 * calling fini inbetween previously allocated memory will be leaked.
 * \param[in,out] msg The previously allocated message pointer.
 * Fields without a default value will not be initialized by this function.
 * You might want to call memset(msg, 0, sizeof(
 * lunar_nav__msg__Crater
 * )) before or use
 * lunar_nav__msg__Crater__create()
 * to allocate and initialize the message.
 * \return true if initialization was successful, otherwise false
 */
ROSIDL_GENERATOR_C_PUBLIC_lunar_nav
bool
lunar_nav__msg__Crater__init(lunar_nav__msg__Crater * msg);

/// Finalize msg/Crater message.
/**
 * \param[in,out] msg The allocated message pointer.
 */
ROSIDL_GENERATOR_C_PUBLIC_lunar_nav
void
lunar_nav__msg__Crater__fini(lunar_nav__msg__Crater * msg);

/// Create msg/Crater message.
/**
 * It allocates the memory for the message, sets the memory to zero, and
 * calls
 * lunar_nav__msg__Crater__init().
 * \return The pointer to the initialized message if successful,
 * otherwise NULL
 */
ROSIDL_GENERATOR_C_PUBLIC_lunar_nav
lunar_nav__msg__Crater *
lunar_nav__msg__Crater__create();

/// Destroy msg/Crater message.
/**
 * It calls
 * lunar_nav__msg__Crater__fini()
 * and frees the memory of the message.
 * \param[in,out] msg The allocated message pointer.
 */
ROSIDL_GENERATOR_C_PUBLIC_lunar_nav
void
lunar_nav__msg__Crater__destroy(lunar_nav__msg__Crater * msg);

/// Check for msg/Crater message equality.
/**
 * \param[in] lhs The message on the left hand size of the equality operator.
 * \param[in] rhs The message on the right hand size of the equality operator.
 * \return true if messages are equal, otherwise false.
 */
ROSIDL_GENERATOR_C_PUBLIC_lunar_nav
bool
lunar_nav__msg__Crater__are_equal(const lunar_nav__msg__Crater * lhs, const lunar_nav__msg__Crater * rhs);

/// Copy a msg/Crater message.
/**
 * This functions performs a deep copy, as opposed to the shallow copy that
 * plain assignment yields.
 *
 * \param[in] input The source message pointer.
 * \param[out] output The target message pointer, which must
 *   have been initialized before calling this function.
 * \return true if successful, or false if either pointer is null
 *   or memory allocation fails.
 */
ROSIDL_GENERATOR_C_PUBLIC_lunar_nav
bool
lunar_nav__msg__Crater__copy(
  const lunar_nav__msg__Crater * input,
  lunar_nav__msg__Crater * output);

/// Initialize array of msg/Crater messages.
/**
 * It allocates the memory for the number of elements and calls
 * lunar_nav__msg__Crater__init()
 * for each element of the array.
 * \param[in,out] array The allocated array pointer.
 * \param[in] size The size / capacity of the array.
 * \return true if initialization was successful, otherwise false
 * If the array pointer is valid and the size is zero it is guaranteed
 # to return true.
 */
ROSIDL_GENERATOR_C_PUBLIC_lunar_nav
bool
lunar_nav__msg__Crater__Sequence__init(lunar_nav__msg__Crater__Sequence * array, size_t size);

/// Finalize array of msg/Crater messages.
/**
 * It calls
 * lunar_nav__msg__Crater__fini()
 * for each element of the array and frees the memory for the number of
 * elements.
 * \param[in,out] array The initialized array pointer.
 */
ROSIDL_GENERATOR_C_PUBLIC_lunar_nav
void
lunar_nav__msg__Crater__Sequence__fini(lunar_nav__msg__Crater__Sequence * array);

/// Create array of msg/Crater messages.
/**
 * It allocates the memory for the array and calls
 * lunar_nav__msg__Crater__Sequence__init().
 * \param[in] size The size / capacity of the array.
 * \return The pointer to the initialized array if successful, otherwise NULL
 */
ROSIDL_GENERATOR_C_PUBLIC_lunar_nav
lunar_nav__msg__Crater__Sequence *
lunar_nav__msg__Crater__Sequence__create(size_t size);

/// Destroy array of msg/Crater messages.
/**
 * It calls
 * lunar_nav__msg__Crater__Sequence__fini()
 * on the array,
 * and frees the memory of the array.
 * \param[in,out] array The initialized array pointer.
 */
ROSIDL_GENERATOR_C_PUBLIC_lunar_nav
void
lunar_nav__msg__Crater__Sequence__destroy(lunar_nav__msg__Crater__Sequence * array);

/// Check for msg/Crater message array equality.
/**
 * \param[in] lhs The message array on the left hand size of the equality operator.
 * \param[in] rhs The message array on the right hand size of the equality operator.
 * \return true if message arrays are equal in size and content, otherwise false.
 */
ROSIDL_GENERATOR_C_PUBLIC_lunar_nav
bool
lunar_nav__msg__Crater__Sequence__are_equal(const lunar_nav__msg__Crater__Sequence * lhs, const lunar_nav__msg__Crater__Sequence * rhs);

/// Copy an array of msg/Crater messages.
/**
 * This functions performs a deep copy, as opposed to the shallow copy that
 * plain assignment yields.
 *
 * \param[in] input The source array pointer.
 * \param[out] output The target array pointer, which must
 *   have been initialized before calling this function.
 * \return true if successful, or false if either pointer
 *   is null or memory allocation fails.
 */
ROSIDL_GENERATOR_C_PUBLIC_lunar_nav
bool
lunar_nav__msg__Crater__Sequence__copy(
  const lunar_nav__msg__Crater__Sequence * input,
  lunar_nav__msg__Crater__Sequence * output);

#ifdef __cplusplus
}
#endif

#endif  // LUNAR_NAV__MSG__DETAIL__CRATER__FUNCTIONS_H_
