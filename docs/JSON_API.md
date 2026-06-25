# KNexus JSON API Documentation

This document describes the schema used to represent HW accelerator device architectures, their properties, and capabilities. The schema is defined in `schema/device_info_schema.json` and is intended for use in device description, capability reporting, and architecture-aware optimization.

## Top-Level Structure

A HW accelerator architecture description is a JSON object with the following required fields:

- **Name**: Architecture name (e.g., "RDNA 3", "Ampere", "Ada Lovelace")
- **Vendor**: Manufacturer (e.g., "AMD", "NVIDIA", "Intel", "ARM", "Qualcomm", "Apple", "Other")
- **Architecture**: Architecture target (e.g., "gfx942", "sm_80")
- **ReleaseYear**: Year the architecture was first released (integer, ≥ 1990)

### Example

```json
{
  "Name": "Ampere",
  "Vendor": "NVIDIA",
  "Architecture": "sm_80",
  "ReleaseYear": 2020,
  ...
}
```

---

## Properties

### FabricationProcess

Describes the semiconductor manufacturing process.

- **ProcessNode** (number, required): Node size in nanometers (e.g., 5, 7, 14)
- **Manufacturer** (string, optional): Foundry name (e.g., "TSMC", "Samsung")
- **Technology** (string, optional): Process technology name (e.g., "N5", "7LPP")

### CoreSubsystem

Describes the compute hierarchy of the chip.

- **Name** (string, optional): Vendor-specific name for compute units (e.g., "CU", "SM", "Xe-core")
- **SubUnits** (array of objects, optional): Sub-units within each compute unit.
  - **Type** (string, required): Type of sub-unit (e.g., "SIMD", "Tensor Core")
  - **Count** (integer, required): Number of these units per parent unit
  - **Size** (integer, required): Number of sub-units inside this unit (e.g., threads)
  - **Description** (string, optional): Description of the sub-unit's function
  - **Memory** (array of strings, optional): Memory types embedded in the core (refer to MemorySubsystem)
  - **SubunitType** (string, optional): Name of the sub-unit (lookup in this list)

### MemorySubsystem

Describes the memory architecture.

- **SupportedMemoryTypes** (array of strings, optional): Types of memory supported (e.g., "GDDR6", "HBM2", "DDR5", etc.)
- **MemoryTypes** (array of objects, optional): Details about each memory level (cache, scratch, registers)
  - **Type** (string, optional): Memory type (e.g., "L1", "L2", "Infinity Cache", "vector register file")
  - **Size** (integer, optional): Size per unit in KB
  - **BankCount** (integer, optional): Number of banks per unit
  - **MaxMemoryBandwidth** (number, optional): Max theoretical bandwidth in GB/s
  - **maxBusWidth** (integer, optional): Max memory bus width in bits
  - **description** (string, optional): Description of the memory/cache

### KernelModel

Describes supported shader/kernel models.

- **LLVMTarget** (string, optional): LLVM target architecture
- **LLVMTriple** (string, optional): LLVM target triple
- **LLVMFeatures** (array of objects, optional): LLVM features
  - **Name** (string, required): Feature name
  - **Description** (string, required): Feature description
- **SubUnits** (array of objects, optional): Subunits within each compute unit
  - **Type** (string, required): Name of the kernel model (e.g., "metal", "opencl", "vulkan")
  - **Version** (string, required): Version of the kernel (e.g., "1.2")

### SpecializedHardware

Describes specialized processing units.

- **RayTracingAccelerators** (object, optional): Ray tracing hardware details
  - **Present** (boolean, optional): Whether dedicated ray tracing hardware is present

---

## Example (Partial)

```json
{
  "Name": "RDNA 3",
  "Vendor": "AMD",
  "Architecture": "gfx1100",
  "ReleaseYear": 2022,
  "FabricationProcess": {
    "ProcessNode": 5,
    "Manufacturer": "TSMC",
    "Technology": "N5"
  },
  "CoreSubsystem": {
    "Name": "CU",
    "SubUnits": [
      {
        "Type": "SIMD",
        "Count": 4,
        "Size": 32,
        "Description": "Vector ALU",
        "Memory": ["vector register file"]
      }
    ]
  },
  "MemorySubsystem": {
    "SupportedMemoryTypes": ["GDDR6", "HBM2"],
    "MemoryTypes": [
      {
        "Type": "L2",
        "Size": 4096,
        "BankCount": 16,
        "MaxMemoryBandwidth": 512,
        "maxBusWidth": 256,
        "description": "Shared L2 cache"
      }
    ]
  },
  "KernelModel": {
    "LLVMTarget": "amdgcn",
    "LLVMTriple": "amdgcn-amd-amdhsa",
    "LLVMFeatures": [
      {"Name": "s-memrealtime", "Description": "Support for real-time memory instructions"}
    ],
    "SubUnits": [
      {"Type": "opencl", "Version": "2.0"}
    ]
  },
  "SpecializedHardware": {
    "RayTracingAccelerators": {
      "Present": true
    }
  }
}
```

---

## Notes

- All fields not marked as required are optional and may be omitted if not applicable.
- The schema is extensible for future hardware features and vendor-specific extensions.
- For a full list of properties and their descriptions, see the JSON schema file: `schema/gpu_architecture_schema.json`.

---

**This schema enables structured, vendor-neutral description of HW accelerator architectures for use in device databases, capability queries, and architecture-aware software.** 