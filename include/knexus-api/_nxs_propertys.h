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
KNEXUS_API_PROP(SubUnits,              _prop_str_vec,    "Sub-Unit Vector")
KNEXUS_API_PROP(SubUnitType,           _prop_str,        "Sub-Unit Type")

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
