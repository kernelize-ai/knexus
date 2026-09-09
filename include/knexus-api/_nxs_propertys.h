/*
 */

/* clang-format off */

#if defined(KNEXUS_API_GENERATE_PROP_ENUM)
/************************************************************************
 * Generate the Function declarations
 ***********************************************************************/
// Generate the Function extern

#define KNEXUS_API_PROP(NAME, TYPE, DESC) \
    NP_##NAME,

enum _nxs_property {

#else
#if defined(KNEXUS_API_GENERATE_PROP_MAP)
/************************************************************************
 * Generate the Property type trait lookup
 ***********************************************************************/
enum _nxs_property_type {
    NPT_INT,
    NPT_FLT,
    NPT_STR,
    NPT_INT_VEC,
    NPT_FLT_VEC,
    NPT_STR_VEC,
    NPT_OBJ_VEC,
    NPT_UNK
};

typedef enum _nxs_property_type nxs_property_type;

#define _prop_int NPT_INT
#define _prop_flt NPT_FLT
#define _prop_str NPT_STR
#define _prop_int_vec NPT_INT_VEC
#define _prop_flt_vec NPT_FLT_VEC
#define _prop_str_vec NPT_STR_VEC
#define _prop_obj_vec NPT_OBJ_VEC

static nxs_property_type nxs_property_type_map[] = {

#define KNEXUS_API_PROP(NAME, TYPE, DESC) \
    TYPE,

#else
#if defined(KNEXUS_API_GENERATE_PROP_TYPE)
/************************************************************************
 * Generate the Property type trait lookup
 ***********************************************************************/
typedef nxs_long _prop_int;
typedef nxs_double _prop_flt;

#ifdef __cplusplus

typedef std::string _prop_str;
typedef std::vector<nxs_long> _prop_int_vec;
typedef std::vector<nxs_double> _prop_flt_vec;
typedef std::vector<std::string> _prop_str_vec;
typedef std::vector<void *> _prop_obj_vec;

template <nxs_property Tnp>
struct nxsPropertyType { typedef void *type; };

#else
typedef char * _prop_str;
typedef void * _prop_int_vec;
typedef void * _prop_flt_vec;
typedef void * _prop_str_vec;
typedef void * _prop_obj_vec;
#endif

// Generate the Property typedefs
#ifdef __cplusplus
#define KNEXUS_API_PROP(NAME, TYPE, DESC) \
    template <> struct nxsPropertyType<NP_##NAME> { typedef TYPE type; };
#else
#define KNEXUS_API_PROP(RETURN_TYPE, NAME, ...)
#endif


#endif
#endif
#endif

/************************************************************************
 * Define API Properties
 ***********************************************************************/

/* THIS SHOULD BE GENERATED FROM THE SCHEMA */

/************************************************************************
 * @def Name
 * @brief Object Name 
 ***********************************************************************/
KNEXUS_API_PROP(Name,                  _prop_str,        "Unit Name")
KNEXUS_API_PROP(Type,                  _prop_str,        "Unit Type")
KNEXUS_API_PROP(Value,                 _prop_int,        "Unit Value")
KNEXUS_API_PROP(ID,                    _prop_int,        "Unit ID")
KNEXUS_API_PROP(Description,           _prop_str,        "Unit Description")

/* Property Hierarchy */
KNEXUS_API_PROP(Count,                 _prop_int,        "Number of Units")
KNEXUS_API_PROP(Size,                  _prop_int,        "Number of Sub-Units")
KNEXUS_API_PROP(Rank,                  _prop_int,        "Rank")
KNEXUS_API_PROP(Shape,                 _prop_int_vec,    "Shape")
KNEXUS_API_PROP(Toolchains,            _prop_obj_vec,    "Supported Kernel Toolchains")
KNEXUS_API_PROP(ChipType,              _prop_str,        "Chip Unit Type Name")
KNEXUS_API_PROP(CoreType,              _prop_str,        "Core Unit Type Name")
KNEXUS_API_PROP(UnitTypes,             _prop_obj_vec,    "Unit Type Map (keyed by type name)")
KNEXUS_API_PROP(Subunits,              _prop_str_vec,    "Contained Sub-Unit Type Names")

KNEXUS_API_PROP(Keys,                  _prop_int_vec,    "Node Keys")

/* Device Properties */
KNEXUS_API_PROP(Vendor,                _prop_str,        "Vendor Name")
KNEXUS_API_PROP(Architecture,          _prop_str,        "Architecture Designation")
KNEXUS_API_PROP(Version,               _prop_str,        "Version String")
KNEXUS_API_PROP(MajorVersion,          _prop_int,        "Major Version")
KNEXUS_API_PROP(MinorVersion,          _prop_int,        "Minor Version")

KNEXUS_API_PROP(CoreSubsystem,         _prop_obj_vec,    "Core Subsystem Hierarchy")
KNEXUS_API_PROP(MemorySubsystem,       _prop_obj_vec,    "Memory Subsystem Hierarchy")

KNEXUS_API_PROP(Limits,                _prop_obj_vec,    "System limits")
KNEXUS_API_PROP(Features,              _prop_str,        "System features")

KNEXUS_API_PROP(GlobalMemorySize,      _prop_int,        "Global memory size (bytes)")
KNEXUS_API_PROP(CoreMemorySize,        _prop_int,        "Core Memory size (bytes)")
KNEXUS_API_PROP(CoreRegisterSize,      _prop_int,        "Core Register size (bytes)")

KNEXUS_API_PROP(SIMDSize,              _prop_int,        "SIMD thread count")

KNEXUS_API_PROP(CoreClockRate,         _prop_int,        "Core clock rate (MHz)")
KNEXUS_API_PROP(MemoryClockRate,       _prop_int,        "Memory clock rate (MHz)")
KNEXUS_API_PROP(MemoryBusWidth,        _prop_int,        "Memory bus width (bits)")

/* Kernel Properties */
KNEXUS_API_PROP(MaxThreadsPerBlock,    _prop_int,        "Max threads per block")
KNEXUS_API_PROP(TimeStamp,             _prop_int,        "Time stamp (cycles)")
KNEXUS_API_PROP(ElapsedTime,           _prop_flt,        "Elapsed time (ms)")

/* Threadgroup Properties */
KNEXUS_API_PROP(MaxThreadsPerThreadgroup, _prop_int,     "Max threads per threadgroup")
KNEXUS_API_PROP(MaxThreadgroupsPerCore,   _prop_int,     "Max threadgroups per Core")
KNEXUS_API_PROP(MaxThreadgroupMemorySize, _prop_int,     "Max threadgroup memory size (bytes)")

KNEXUS_API_PROP(Location,              _prop_str,        "Location")
KNEXUS_API_PROP(MaxTransferRate,       _prop_int,        "Max transfer rate (bytes/sec)")
KNEXUS_API_PROP(UnifiedMemory,         _prop_int,        "Unified Memory present")
KNEXUS_API_PROP(MaxBufferSize,         _prop_int,        "Max buffer size (bytes)")

KNEXUS_API_PROP(DataTypes,             _prop_str_vec,    "Data Types")
KNEXUS_API_PROP(ClockModes,            _prop_str_vec,    "Clock Modes")
KNEXUS_API_PROP(BaseClock,             _prop_int,        "Base Clock (MHz)")
KNEXUS_API_PROP(PowerModes,            _prop_str_vec,    "Power Modes")
KNEXUS_API_PROP(MaxPower,              _prop_flt,        "Max Power")

KNEXUS_API_PROP(CoreUtilization,      _prop_int,          "Core Utilization")
KNEXUS_API_PROP(MemoryUtilization,    _prop_int,          "Memory Utilization")

/* Performance Properties
 *
 * The speeds-and-feeds vocabulary of schema/device_info_schema.json
 * (#/definitions/Performance, documented in docs/JSON_API.md), carried by every
 * CoreSubsystem.UnitTypes entry and every MemorySubsystem.MemoryTypes entry. Both are
 * name-keyed maps, so a path to a figure names its unit or its memory rather than
 * numbering it: {"MemorySubsystem", "MemoryTypes", "HBM3", "Performance",
 * "TransferRate"}. NP_Keys lists either map's names.
 *
 * Every numeric field below MUST have a row here, and the type is a decision, not a
 * transcription of whatever the first device file happened to write. InfoImpl::getProp
 * (src/info.cpp) uses nxsGetPropEnum() on the last path segment only as a hint; on a miss
 * it falls through to getNodeType(), which answers NPT_INT for a whole-number JSON literal
 * and NPT_FLT for a fractional one. So an *undeclared* numeric field is an int on one
 * device file and a double on the next, purely from how the number was written -- and a
 * fractional value asked for as an int is silently truncated. Rate, ClockRate,
 * TransferRate and Latency are quantities that can be fractional, so they are _prop_flt
 * and read back as doubles even from a literal such as 1350 or 664. LaneWidth counts
 * discrete bits, so it is _prop_int. test/cpp/test_property_types.cpp pins all of it.
 *
 * The throughput figure is named Rate, not Value: NP_Value is already _prop_int and is
 * used as an integer by the runtime plugins (plugins/{cpu,cuda,tenstorrent}/*_runtime.cpp),
 * so a Throughput "Value" of 5.4 would be truncated to 5.
 */
KNEXUS_API_PROP(Performance,          _prop_obj_vec,      "Speeds and feeds of one unit")
KNEXUS_API_PROP(ClockRate,            _prop_flt,          "Clock rate of one unit (MHz)")
KNEXUS_API_PROP(LaneWidth,            _prop_int,          "Width of one SIMD lane (bits)")
KNEXUS_API_PROP(TransferRate,         _prop_flt,          "Interface signalling rate (GT/s)")
KNEXUS_API_PROP(Latency,              _prop_flt,          "Access latency (cycles)")
KNEXUS_API_PROP(Source,               _prop_str,          "Figure provenance (Published/Derived/Measured)")
KNEXUS_API_PROP(Throughput,           _prop_obj_vec,      "Rate figures, one per precision and mode")
KNEXUS_API_PROP(Rate,                 _prop_flt,          "Throughput figure, expressed in Unit")
KNEXUS_API_PROP(Unit,                 _prop_str,          "Unit of Rate (e.g. TFLOP/s, GB/s)")
KNEXUS_API_PROP(Precision,            _prop_str,          "Numeric format Rate applies to")
KNEXUS_API_PROP(Mode,                 _prop_str,          "Operating mode Rate applies to")

/************************************************************************
 * Cleanup
 ***********************************************************************/
#ifdef KNEXUS_API_GENERATE_PROP_ENUM
    NXS_PROPERTY_CNT,
    NXS_PROPERTY_PREFIX_LEN        = 3,

    NXS_PROPERTY_INVALID = -1
}; /* close _nxs_property */

typedef enum _nxs_property nxs_property;

/* Translation functions */
nxs_int nxsGetPropCount();
const char *nxsGetPropName(nxs_int propEnum);
nxs_property nxsGetPropEnum(const char *propName);

const char *nxsGetStatusName(nxs_int statusEnum);
nxs_status nxsGetStatusEnum(const char *statusName);
#else
#if defined(KNEXUS_API_GENERATE_PROP_MAP)

}; /* close nxs_property_type_map */

#undef _prop_int
#undef _prop_flt
#undef _prop_str
#undef _prop_int_vec
#undef _prop_flt_vec
#undef _prop_str_vec
#undef _prop_obj_vec

#endif
#endif

/* clang-format off */

#undef KNEXUS_API_GENERATE_PROP_ENUM
#undef KNEXUS_API_GENERATE_PROP_MAP
#undef KNEXUS_API_GENERATE_PROP_TYPE

#undef _KNEXUS_API_PROP
#undef KNEXUS_API_PROP
