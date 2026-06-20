#include <pybind11/pybind11.h>

#include "pyknexus.h"

#define STRINGIFY(x) #x
#define MACRO_STRINGIFY(x) STRINGIFY(x)

namespace py = pybind11;

PYBIND11_MODULE(libknexus, m) {
  m.doc() = R"pbdoc(
        KNexus - Python API
        -----------------------

        .. currentmodule:: knexus

        .. autosummary::
           :toctree: _generate

           system
           devices
    )pbdoc";

  // remove extra 'system' module (its redundant)
  pyknexus::init_system_bindings(m);

#ifdef VERSION_INFO
  m.attr("__version__") = MACRO_STRINGIFY(VERSION_INFO);
#else
  m.attr("__version__") = "dev";
#endif
}
