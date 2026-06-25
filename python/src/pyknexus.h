#ifndef PYKNEXUS_H
#define PYKNEXUS_H

#include <knexus-api.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

namespace py = pybind11;

namespace pyknexus {

void init_system_bindings(py::module &m);

}  // namespace pyknexus

#endif  // PYKNEXUS_H
