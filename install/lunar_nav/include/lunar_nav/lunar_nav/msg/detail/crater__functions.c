// generated from rosidl_generator_c/resource/idl__functions.c.em
// with input from lunar_nav:msg/Crater.idl
// generated code does not contain a copyright notice
#include "lunar_nav/msg/detail/crater__functions.h"

#include <assert.h>
#include <stdbool.h>
#include <stdlib.h>
#include <string.h>

#include "rcutils/allocator.h"


bool
lunar_nav__msg__Crater__init(lunar_nav__msg__Crater * msg)
{
  if (!msg) {
    return false;
  }
  // x
  // y
  // z
  // radius
  // depth
  // confidence
  // id
  return true;
}

void
lunar_nav__msg__Crater__fini(lunar_nav__msg__Crater * msg)
{
  if (!msg) {
    return;
  }
  // x
  // y
  // z
  // radius
  // depth
  // confidence
  // id
}

bool
lunar_nav__msg__Crater__are_equal(const lunar_nav__msg__Crater * lhs, const lunar_nav__msg__Crater * rhs)
{
  if (!lhs || !rhs) {
    return false;
  }
  // x
  if (lhs->x != rhs->x) {
    return false;
  }
  // y
  if (lhs->y != rhs->y) {
    return false;
  }
  // z
  if (lhs->z != rhs->z) {
    return false;
  }
  // radius
  if (lhs->radius != rhs->radius) {
    return false;
  }
  // depth
  if (lhs->depth != rhs->depth) {
    return false;
  }
  // confidence
  if (lhs->confidence != rhs->confidence) {
    return false;
  }
  // id
  if (lhs->id != rhs->id) {
    return false;
  }
  return true;
}

bool
lunar_nav__msg__Crater__copy(
  const lunar_nav__msg__Crater * input,
  lunar_nav__msg__Crater * output)
{
  if (!input || !output) {
    return false;
  }
  // x
  output->x = input->x;
  // y
  output->y = input->y;
  // z
  output->z = input->z;
  // radius
  output->radius = input->radius;
  // depth
  output->depth = input->depth;
  // confidence
  output->confidence = input->confidence;
  // id
  output->id = input->id;
  return true;
}

lunar_nav__msg__Crater *
lunar_nav__msg__Crater__create()
{
  rcutils_allocator_t allocator = rcutils_get_default_allocator();
  lunar_nav__msg__Crater * msg = (lunar_nav__msg__Crater *)allocator.allocate(sizeof(lunar_nav__msg__Crater), allocator.state);
  if (!msg) {
    return NULL;
  }
  memset(msg, 0, sizeof(lunar_nav__msg__Crater));
  bool success = lunar_nav__msg__Crater__init(msg);
  if (!success) {
    allocator.deallocate(msg, allocator.state);
    return NULL;
  }
  return msg;
}

void
lunar_nav__msg__Crater__destroy(lunar_nav__msg__Crater * msg)
{
  rcutils_allocator_t allocator = rcutils_get_default_allocator();
  if (msg) {
    lunar_nav__msg__Crater__fini(msg);
  }
  allocator.deallocate(msg, allocator.state);
}


bool
lunar_nav__msg__Crater__Sequence__init(lunar_nav__msg__Crater__Sequence * array, size_t size)
{
  if (!array) {
    return false;
  }
  rcutils_allocator_t allocator = rcutils_get_default_allocator();
  lunar_nav__msg__Crater * data = NULL;

  if (size) {
    data = (lunar_nav__msg__Crater *)allocator.zero_allocate(size, sizeof(lunar_nav__msg__Crater), allocator.state);
    if (!data) {
      return false;
    }
    // initialize all array elements
    size_t i;
    for (i = 0; i < size; ++i) {
      bool success = lunar_nav__msg__Crater__init(&data[i]);
      if (!success) {
        break;
      }
    }
    if (i < size) {
      // if initialization failed finalize the already initialized array elements
      for (; i > 0; --i) {
        lunar_nav__msg__Crater__fini(&data[i - 1]);
      }
      allocator.deallocate(data, allocator.state);
      return false;
    }
  }
  array->data = data;
  array->size = size;
  array->capacity = size;
  return true;
}

void
lunar_nav__msg__Crater__Sequence__fini(lunar_nav__msg__Crater__Sequence * array)
{
  if (!array) {
    return;
  }
  rcutils_allocator_t allocator = rcutils_get_default_allocator();

  if (array->data) {
    // ensure that data and capacity values are consistent
    assert(array->capacity > 0);
    // finalize all array elements
    for (size_t i = 0; i < array->capacity; ++i) {
      lunar_nav__msg__Crater__fini(&array->data[i]);
    }
    allocator.deallocate(array->data, allocator.state);
    array->data = NULL;
    array->size = 0;
    array->capacity = 0;
  } else {
    // ensure that data, size, and capacity values are consistent
    assert(0 == array->size);
    assert(0 == array->capacity);
  }
}

lunar_nav__msg__Crater__Sequence *
lunar_nav__msg__Crater__Sequence__create(size_t size)
{
  rcutils_allocator_t allocator = rcutils_get_default_allocator();
  lunar_nav__msg__Crater__Sequence * array = (lunar_nav__msg__Crater__Sequence *)allocator.allocate(sizeof(lunar_nav__msg__Crater__Sequence), allocator.state);
  if (!array) {
    return NULL;
  }
  bool success = lunar_nav__msg__Crater__Sequence__init(array, size);
  if (!success) {
    allocator.deallocate(array, allocator.state);
    return NULL;
  }
  return array;
}

void
lunar_nav__msg__Crater__Sequence__destroy(lunar_nav__msg__Crater__Sequence * array)
{
  rcutils_allocator_t allocator = rcutils_get_default_allocator();
  if (array) {
    lunar_nav__msg__Crater__Sequence__fini(array);
  }
  allocator.deallocate(array, allocator.state);
}

bool
lunar_nav__msg__Crater__Sequence__are_equal(const lunar_nav__msg__Crater__Sequence * lhs, const lunar_nav__msg__Crater__Sequence * rhs)
{
  if (!lhs || !rhs) {
    return false;
  }
  if (lhs->size != rhs->size) {
    return false;
  }
  for (size_t i = 0; i < lhs->size; ++i) {
    if (!lunar_nav__msg__Crater__are_equal(&(lhs->data[i]), &(rhs->data[i]))) {
      return false;
    }
  }
  return true;
}

bool
lunar_nav__msg__Crater__Sequence__copy(
  const lunar_nav__msg__Crater__Sequence * input,
  lunar_nav__msg__Crater__Sequence * output)
{
  if (!input || !output) {
    return false;
  }
  if (output->capacity < input->size) {
    const size_t allocation_size =
      input->size * sizeof(lunar_nav__msg__Crater);
    rcutils_allocator_t allocator = rcutils_get_default_allocator();
    lunar_nav__msg__Crater * data =
      (lunar_nav__msg__Crater *)allocator.reallocate(
      output->data, allocation_size, allocator.state);
    if (!data) {
      return false;
    }
    // If reallocation succeeded, memory may or may not have been moved
    // to fulfill the allocation request, invalidating output->data.
    output->data = data;
    for (size_t i = output->capacity; i < input->size; ++i) {
      if (!lunar_nav__msg__Crater__init(&output->data[i])) {
        // If initialization of any new item fails, roll back
        // all previously initialized items. Existing items
        // in output are to be left unmodified.
        for (; i-- > output->capacity; ) {
          lunar_nav__msg__Crater__fini(&output->data[i]);
        }
        return false;
      }
    }
    output->capacity = input->size;
  }
  output->size = input->size;
  for (size_t i = 0; i < input->size; ++i) {
    if (!lunar_nav__msg__Crater__copy(
        &(input->data[i]), &(output->data[i])))
    {
      return false;
    }
  }
  return true;
}
