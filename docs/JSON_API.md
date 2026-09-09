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
- **ChipType** (string, optional): Unit type representing the whole chip (a key in `UnitTypes`)
- **CoreType** (string, optional): Unit type representing a single core (a key in `UnitTypes`, e.g., `"NeuronCore-v3"`, `"Tensix Core"`)
- **UnitTypes** (object, optional): Unit types in the compute hierarchy, keyed by type name (e.g., "SIMD", "Tensor Core"). Each value describes one unit type:
  - **Count** (integer, required): Number of these units per parent unit
  - **Size** (integer, optional, default `1`): Number of sub-units inside this unit (e.g., threads)
  - **Description** (string, optional): Description of the unit's function
  - **Memory** (array of strings, optional): Memory types embedded in the core (each entry is a key in `MemorySubsystem.MemoryTypes`, matched verbatim)
  - **Subunits** (array of strings, optional): Names of the unit types contained in this unit (each entry is a key in `UnitTypes`)
  - **Performance** (object, optional): Speeds and feeds of **one** unit of this type - see [Performance](#performance)

### MemorySubsystem

Describes the memory architecture.

- **SupportedMemoryTypes** (array of strings, optional): Types of memory supported (e.g., "GDDR6", "HBM2", "DDR5", etc.)
- **MemoryTypes** (object, optional): Details about each memory level (cache, scratch, registers),
  keyed by memory type name (e.g., `"L2"`, `"HBM3"`, `"LDS"`, `"VGPR"`, `"Tensix L1"`) exactly
  as `CoreSubsystem.UnitTypes` is keyed by unit type name. The key *is* the memory's type, so a
  memory is addressed by name -- `MemorySubsystem/MemoryTypes/L2/Size` -- and `UnitTypes.*.Memory`
  entries name a key.

  **The key is an identifier, not a caption.** It is the memory's short name, preferring the
  established acronym where the domain has one (`VGPR`, `SGPR`, `LDS`, `SBUF`, `PSUM`, `SLC`,
  `L1`, `L2`, `HBM3`). It carries **no parenthetical scope gloss and no `/`** -- write the scope
  in `description`, which is where a reader looks for it and where it can be stated in a
  sentence, and which keeps every JSON pointer over the key free of RFC 6901 escaping. Keys must
  be unique within a file (JSON guarantees this) and should name the same concept the same way
  across files, because comparing one device against another is a first-class use of this
  library. Where shortening two memories to the same name would collide -- Tenstorrent declares
  an L1 SRAM in both its Tensix and its Ethernet tiles -- disambiguate with the minimal
  distinguishing scope word (`"Tensix L1"`, `"Ethernet L1"`), never by restoring the full gloss.
  `test/python/test_device_schema.py`'s `memory-key` check enforces the punctuation rule.

  Each value describes one memory level:
  - **Size** (integer, optional): Size per unit in KB
  - **BankCount** (integer, optional): Number of banks per unit
  - **MaxMemoryBandwidth** (number, optional): Max theoretical bandwidth in GB/s
  - **maxBusWidth** (integer, optional): Max memory bus width in bits
  - **description** (string, optional): Description of the memory/cache
  - **Performance** (object, optional): Speeds and feeds of **one** instance of this memory - see [Performance](#performance)

### Performance

The machine-readable speeds and feeds of a `CoreSubsystem.UnitTypes` entry or a
`MemorySubsystem.MemoryTypes` entry. Both carry the same block (`#/definitions/Performance`
in the schema). It is additive and optional: an entry with no figure that is published,
derivable or measured omits `Performance` entirely rather than guessing one.

Four rules govern the block:

1. **Every figure is per *one* unit** - one core, one engine, one link, one instance of that
   memory. The entry's `Count` supplies the multiplier, so chip totals are not restated. A
   `Chip` entry is itself a unit with `Count: 1`, which is where a genuinely chip-scoped
   figure belongs.
2. **Every figure carries a `Source`** - `"Published"`, `"Derived"` or `"Measured"`. The
   scalar fields share the block's `Source`, and the schema *requires* it as soon as any of
   them is present; each `Throughput` entry carries its own.
3. **Throughput is an array, not fixed keys.** Fixed keys such as `fp16TopsPerGPU` cannot
   express Tenstorrent's LoFi/HiFi2/HiFi4 fidelity phases or NVIDIA's dense/sparse pairs, and
   a new key per precision is unbounded.
4. **Nothing here is inferred at read time.** Every field below has a `KNEXUS_API_PROP` row in
   `include/knexus-api/_nxs_propertys.h`, so its C++ type is fixed by declaration. Without a
   row, `InfoImpl::getProp` falls back to the JSON literal's shape and the same field reads
   back as an integer on one device file and a double on the next. `ClockRate` and `Rate` are
   declared `_prop_flt`: `"ClockRate": 1350` reads back as `1350.0`, not as `1350`.

Fields (all optional, but the block must not be empty):

- **ClockRate** (number, MHz): Clock this unit runs at, and the clock the `Throughput` figures
  are quoted at (e.g. `1350` for Blackhole's 1.35 GHz AI clock, `1980` for an H100 SXM5 boost clock)
- **LaneWidth** (integer, bits): Width of **one** SIMD lane of this unit (e.g. `32` for the
  Blackhole SFPU's "32 lanes of 32 bits wide", `64` for a Hopper FP64 core). The lane count is
  the entry's `Size` and is not restated here
- **TransferRate** (number, GT/s): Signalling rate of this unit's external interface, equivalently
  Gbps per pin or per lane (e.g. `16` for Blackhole's GDDR6, `5.2` for an MI300X HBM3 stack,
  `6.4` for LPDDR5 at 6400 MT/s). Times the bus width in bits, over 8, gives GB/s
- **Latency** (number, cycles): Access latency (e.g. `2` for the 2-cycle load latency of a
  Blackhole Private RISC-V Data RAM)
- **Source** (string): Provenance of the scalar fields above - `"Published"` | `"Derived"` |
  `"Measured"`. Required whenever any of them is present
- **Throughput** (array of objects): Rate figures, one entry per precision and operating mode
  - **Rate** (number, required): The figure itself, in `Unit`
  - **Unit** (string, required): One of `"TFLOP/s"`, `"GFLOP/s"`, `"TOP/s"`, `"GOP/s"`, `"GB/s"`,
    `"Gb/s"`, `"MAC/cycle"`, `"FLOP/cycle"`, `"OP/cycle"`, `"instruction/cycle"`, `"byte/cycle"`.
    A closed vocabulary, so two devices' figures are comparable; a missing unit earns a new enum
    value rather than a free string
  - **Source** (string, required): `"Published"` | `"Derived"` | `"Measured"`
  - **Precision** (string, optional): `"FP64"`, `"FP32"`, `"TF32"`, `"BF16"`, `"FP16"`, `"FP8"`,
    `"BFP8"`, `"BFP4"`, `"BFP2"`, `"INT32"`, `"INT8"`, `"INT4"`, `"Other"`. Omit for rates that are
    precision-independent, such as a link rate or an instruction issue rate
  - **Mode** (string, optional): The operating mode the rate applies to, where one unit has more
    than one. Vendor-specific by nature, so this is a free string: e.g. `"LoFi"`, `"HiFi2"`,
    `"HiFi4"`, `"Dense"`, `"Sparse"`, `"Bidirectional"`, `"Read"`, `"Write"`, `"Fused"`

`MaxMemoryBandwidth` on a `MemoryTypes` entry remains the plain theoretical peak in GB/s. Add a
`Throughput` entry alongside it only for what that field cannot say - a provenance, a direction
or mode, a measured value, or a different unit. Where both state the same quantity they must agree.

```json
"Matrix Unit (FPU)": {
  "Count": 1,
  "Size": 1,
  "Description": "Per-Tensix matrix engine ...",
  "Performance": {
    "ClockRate": 1350,
    "Source": "Published",
    "Throughput": [
      {"Rate": 4096, "Unit": "MAC/cycle", "Source": "Published"},
      {"Precision": "BFP8", "Mode": "LoFi",  "Rate": 5.4,  "Unit": "TFLOP/s", "Source": "Measured"},
      {"Precision": "BFP8", "Mode": "HiFi2", "Rate": 2.7,  "Unit": "TFLOP/s", "Source": "Measured"},
      {"Precision": "BFP8", "Mode": "HiFi4", "Rate": 1.35, "Unit": "TFLOP/s", "Source": "Measured"}
    ]
  }
}
```

### KernelModel

Describes supported shader/kernel models.

- **LLVMTarget** (string, optional): LLVM target architecture
- **LLVMTriple** (string, optional): LLVM target triple
- **LLVMFeatures** (array of objects, optional): LLVM features
  - **Name** (string, required): Feature name
  - **Description** (string, required): Feature description
- **Toolchains** (array of objects, optional): Kernel toolchains this device supports - the
  programming models, SDKs, compilers and shading languages a kernel can be written against
  - **Type** (string, required): Name of the toolchain, programming model or shading language (e.g., "CUDA", "HIP", "Metal", "OpenCL", "PTX ISA", "TT-Metalium")
  - **Version** (string, required): Supported version of that toolchain (e.g., "12.8", "3.1", "0.76.0")

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
    "CoreType": "SIMD",
    "UnitTypes": {
      "SIMD": {
        "Count": 4,
        "Size": 32,
        "Description": "Vector ALU",
        "Memory": ["VGPR"],
        "Performance": {
          "ClockRate": 2500,
          "LaneWidth": 32,
          "Source": "Published",
          "Throughput": [
            {"Precision": "FP32", "Rate": 2, "Unit": "FLOP/cycle", "Source": "Published"}
          ]
        }
      }
    }
  },
  "MemorySubsystem": {
    "SupportedMemoryTypes": ["GDDR6", "HBM2"],
    "MemoryTypes": {
      "VGPR": {
        "Size": 192,
        "description": "Per-SIMD vector general-purpose registers"
      },
      "L2": {
        "Size": 4096,
        "BankCount": 16,
        "MaxMemoryBandwidth": 512,
        "maxBusWidth": 256,
        "description": "Shared L2 cache",
        "Performance": {
          "TransferRate": 16,
          "Source": "Published",
          "Throughput": [
            {"Rate": 512, "Unit": "GB/s", "Source": "Derived"}
          ]
        }
      }
    }
  },
  "KernelModel": {
    "LLVMTarget": "amdgcn",
    "LLVMTriple": "amdgcn-amd-amdhsa",
    "LLVMFeatures": [
      {"Name": "s-memrealtime", "Description": "Support for real-time memory instructions"}
    ],
    "Toolchains": [
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
- For a full list of properties and their descriptions, see the JSON schema file: `schema/device_info_schema.json`.

---

**This schema enables structured, vendor-neutral description of HW accelerator architectures for use in device databases, capability queries, and architecture-aware software.** 