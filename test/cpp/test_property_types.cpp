// Property types are DECLARED, not inferred.
//
// InfoImpl::getProp (src/info.cpp) resolves the last segment of a property path through
// nxsGetPropEnum(). On a hit it takes the type from nxs_property_type_map, i.e. from the
// KNEXUS_API_PROP row in include/knexus-api/_nxs_propertys.h. On a MISS it falls through to
// InfoImpl::getNodeType(), which answers NPT_INT for a whole-number JSON literal and NPT_FLT
// for a fractional one -- so an undeclared numeric field is an int on one device file and a
// double on the next, purely from how the number happened to be written.
//
// This test pins that difference for the speeds-and-feeds vocabulary added to
// schema/device_info_schema.json: a field declared _prop_flt must read back as nxs_double
// even when the JSON literal is a whole number (`"ClockRate": 1350`, not `1350.0`), and the
// control group shows the same literal reading back as nxs_long when no row declares it.
//
// It also pins the keyed MemorySubsystem.MemoryTypes map: a memory is addressed by name
// through InfoImpl::getNode's object-member lookup (no C++ change was needed for that), and
// "Keys" -- unanswerable for the array this used to be -- now lists the memory names.
//
// The fixture's two memory keys are DELIBERATELY shapes device_lib/ no longer uses. The
// corpus now keys every memory by a succinct identifier -- "L2", "GDDR6", "Tensix L1" --
// with no '(' and no '/', a data convention the schema validator's memory-key check
// enforces (test/python/test_device_schema.py). Nothing at THIS layer enforces it: the
// schema types MemoryTypes as additionalProperties over arbitrary strings, and a caller
// may build an Info document of its own. So the C++ contract is broader than the corpus
// convention, and only a fixture that is not corpus-shaped can pin it:
//
//   * "L2 / System Level Cache" proves getNode does an nlohmann MEMBER LOOKUP, not a JSON
//     Pointer resolve. If it ever resolved pointers, the '/' would split the key into two
//     hops and this lookup would fail -- and RFC 6901 escaping would become a caller's
//     problem for every key. That distinction is invisible once every key is punctuation
//     free, which is exactly why the test keeps a key that is not.
//   * "GDDR6 (Off-chip DRAM)" is a strict superstring of the "GDDR6" the corpus now uses,
//     so the negative assertion below proves lookup is exact rather than prefix or fuzzy.
//
// Retiring this coverage would mean deleting the only runtime proof of both properties, so
// the fixture keeps the awkward keys on purpose rather than tracking the corpus rename.
//
// It needs no runtime, no device, no plugin and no command-line argument: knexus::Info reads
// a JSON file directly. The fixture is written to a temp file by the test itself, so nothing
// in device_lib/ is involved.

#include <gtest/gtest.h>
#include <knexus-api.h>
#include <knexus/info.h>

#include <cstdio>
#include <filesystem>
#include <fstream>
#include <string>
#include <variant>
#include <vector>

namespace {

// One device object shaped exactly like a backfilled device_lib entry. Every number that
// carries a declared type is written as a WHOLE-NUMBER literal on purpose -- that is the
// shape that makes an undeclared field come back as an integer.
constexpr const char *kFixture = R"JSON({
  "Name": "PropertyTypeFixture",
  "Vendor": "Other",
  "Architecture": "fixture",
  "ReleaseYear": 2024,
  "CoreSubsystem": {
    "ChipType": "Chip",
    "UnitTypes": {
      "Matrix Unit (FPU)": {
        "Count": 1,
        "Size": 1,
        "Performance": {
          "ClockRate": 1350,
          "LaneWidth": 32,
          "Source": "Published",
          "Throughput": [
            {"Rate": 664, "Unit": "TFLOP/s", "Precision": "BFP8", "Mode": "LoFi",
             "Source": "Published"},
            {"Rate": 5.4, "Unit": "TFLOP/s", "Precision": "BFP8", "Mode": "HiFi2",
             "Source": "Measured"}
          ]
        }
      }
    }
  },
  "MemorySubsystem": {
    "MemoryTypes": {
      "L2 / System Level Cache": {
        "Performance": {
          "TransferRate": 4,
          "Latency": 1,
          "Source": "Published",
          "Throughput": [{"Rate": 64, "Unit": "GB/s", "Source": "Derived"}]
        }
      },
      "GDDR6 (Off-chip DRAM)": {
        "Performance": {
          "TransferRate": 16,
          "Latency": 2,
          "Source": "Published",
          "Throughput": [{"Rate": 512, "Unit": "GB/s", "Source": "Derived"}]
        }
      }
    }
  },
  "ControlGroup": {
    "NotADeclaredProperty": 1350,
    "AlsoNotADeclaredProperty": 1350.5
  }
})JSON";

// Path prefixes into the fixture, spelled once.
const std::vector<std::string_view> kUnitPerf = {
    "CoreSubsystem", "UnitTypes", "Matrix Unit (FPU)", "Performance"};
// MemorySubsystem.MemoryTypes is a name-keyed map (as CoreSubsystem.UnitTypes is), so a
// memory is reached by its name -- never by the position it happens to sit at.
const std::vector<std::string_view> kMemTypes = {"MemorySubsystem", "MemoryTypes"};
const std::vector<std::string_view> kMemPerf = {
    "MemorySubsystem", "MemoryTypes", "GDDR6 (Off-chip DRAM)", "Performance"};
const std::vector<std::string_view> kCachePerf = {
    "MemorySubsystem", "MemoryTypes", "L2 / System Level Cache", "Performance"};

std::vector<std::string_view> Path(const std::vector<std::string_view> &prefix,
                                   std::initializer_list<std::string_view> tail) {
  std::vector<std::string_view> path(prefix);
  path.insert(path.end(), tail);
  return path;
}

// std::holds_alternative / std::get want the std::variant itself; knexus::Property derives
// from it. Cast explicitly rather than lean on derived-to-base template deduction.
const knexus::PropVariant &AsVariant(const knexus::Property &property) {
  return static_cast<const knexus::PropVariant &>(property);
}

// A fixture file that deletes itself, so the test leaves nothing behind.
class PropertyTypes : public ::testing::Test {
 protected:
  static void SetUpTestSuite() {
    path_ = std::filesystem::temp_directory_path() /
            ("knexus_property_types_fixture.json");
    std::ofstream out(path_);
    out << kFixture;
    out.close();
    info_ = new knexus::Info(path_.string());
  }

  static void TearDownTestSuite() {
    delete info_;
    info_ = nullptr;
    std::error_code ignored;
    std::filesystem::remove(path_, ignored);
  }

  static knexus::Info &Device() { return *info_; }

  static std::filesystem::path path_;
  static knexus::Info *info_;
};

std::filesystem::path PropertyTypes::path_;
knexus::Info *PropertyTypes::info_ = nullptr;

// ---------------------------------------------------------------------------------------
// the table itself
// ---------------------------------------------------------------------------------------

TEST_F(PropertyTypes, EveryNewFieldHasAPropertyRow) {
  for (const char *name : {"Performance", "ClockRate", "LaneWidth", "TransferRate",
                           "Latency", "Source", "Throughput", "Rate", "Unit",
                           "Precision", "Mode"}) {
    EXPECT_NE(nxsGetPropEnum(name), NXS_PROPERTY_INVALID)
        << name << " has no KNEXUS_API_PROP row, so its type would be inferred from the "
                   "JSON literal";
  }
}

TEST_F(PropertyTypes, DeclaredTypesAreTheIntendedOnes) {
  EXPECT_EQ(nxs_property_type_map[NP_ClockRate], NPT_FLT);
  EXPECT_EQ(nxs_property_type_map[NP_Rate], NPT_FLT);
  EXPECT_EQ(nxs_property_type_map[NP_TransferRate], NPT_FLT);
  EXPECT_EQ(nxs_property_type_map[NP_Latency], NPT_FLT);
  EXPECT_EQ(nxs_property_type_map[NP_LaneWidth], NPT_INT);
  EXPECT_EQ(nxs_property_type_map[NP_Source], NPT_STR);
  EXPECT_EQ(nxs_property_type_map[NP_Unit], NPT_STR);
  EXPECT_EQ(nxs_property_type_map[NP_Precision], NPT_STR);
  EXPECT_EQ(nxs_property_type_map[NP_Mode], NPT_STR);
  EXPECT_EQ(nxs_property_type_map[NP_Performance], NPT_OBJ_VEC);
  EXPECT_EQ(nxs_property_type_map[NP_Throughput], NPT_OBJ_VEC);

  // Rate exists because Value cannot be reused: it is already an integer, and the runtime
  // plugins read it as one. A Throughput "Value" of 5.4 would be truncated to 5.
  EXPECT_EQ(nxs_property_type_map[NP_Value], NPT_INT);
}

// ---------------------------------------------------------------------------------------
// the whole point: a whole-number literal in a declared float field reads back as a double
// ---------------------------------------------------------------------------------------

TEST_F(PropertyTypes, WholeNumberLiteralInADeclaredFloatFieldReadsBackAsDouble) {
  // "ClockRate": 1350 -- no decimal point anywhere in the JSON.
  auto clock = Device().getProperty(Path(kUnitPerf, {"ClockRate"}));
  ASSERT_TRUE(clock.has_value()) << "CoreSubsystem/UnitTypes/.../Performance/ClockRate";
  EXPECT_TRUE(std::holds_alternative<nxs_double>(AsVariant(*clock)))
      << "ClockRate is declared _prop_flt but came back as index "
      << AsVariant(*clock).index();
  EXPECT_FALSE(std::holds_alternative<nxs_long>(AsVariant(*clock)));
  EXPECT_DOUBLE_EQ(std::get<nxs_double>(AsVariant(*clock)), 1350.0);
}

TEST_F(PropertyTypes, ThroughputRateIsADoubleWhetherOrNotTheLiteralHasAPoint) {
  // Element 0 is `"Rate": 664`, element 1 is `"Rate": 5.4`. One declared type, both shapes.
  auto whole = Device().getProperty(Path(kUnitPerf, {"Throughput", "0", "Rate"}));
  ASSERT_TRUE(whole.has_value());
  EXPECT_TRUE(std::holds_alternative<nxs_double>(AsVariant(*whole)));
  EXPECT_DOUBLE_EQ(std::get<nxs_double>(AsVariant(*whole)), 664.0);

  auto fractional = Device().getProperty(Path(kUnitPerf, {"Throughput", "1", "Rate"}));
  ASSERT_TRUE(fractional.has_value());
  EXPECT_TRUE(std::holds_alternative<nxs_double>(AsVariant(*fractional)));
  EXPECT_DOUBLE_EQ(std::get<nxs_double>(AsVariant(*fractional)), 5.4);

  // Same type on both, which is the property an undeclared field cannot offer.
  EXPECT_EQ(AsVariant(*whole).index(), AsVariant(*fractional).index());
}

TEST_F(PropertyTypes, MemoryPerformanceFieldsCarryTheirDeclaredTypes) {
  auto transfer = Device().getProperty(Path(kMemPerf, {"TransferRate"}));  // literal: 16
  ASSERT_TRUE(transfer.has_value());
  EXPECT_TRUE(std::holds_alternative<nxs_double>(AsVariant(*transfer)));
  EXPECT_DOUBLE_EQ(std::get<nxs_double>(AsVariant(*transfer)), 16.0);

  auto latency = Device().getProperty(Path(kMemPerf, {"Latency"}));  // literal: 2
  ASSERT_TRUE(latency.has_value());
  EXPECT_TRUE(std::holds_alternative<nxs_double>(AsVariant(*latency)));
  EXPECT_DOUBLE_EQ(std::get<nxs_double>(AsVariant(*latency)), 2.0);

  auto rate = Device().getProperty(Path(kMemPerf, {"Throughput", "0", "Rate"}));  // 512
  ASSERT_TRUE(rate.has_value());
  EXPECT_TRUE(std::holds_alternative<nxs_double>(AsVariant(*rate)));
  EXPECT_DOUBLE_EQ(std::get<nxs_double>(AsVariant(*rate)), 512.0);
}

// ---------------------------------------------------------------------------------------
// the keyed map: a memory is reached by name, and the map answers "Keys"
// ---------------------------------------------------------------------------------------

TEST_F(PropertyTypes, AMemoryIsAddressedByNameNotByPosition) {
  // Both memories declare TransferRate. Only the name distinguishes them, and the name is
  // the map key -- "L2 / System Level Cache" is a member lookup, not a JSON pointer, so the
  // '/' in it needs no escaping. No device_lib/ key has a '/' any more; the fixture keeps
  // one precisely because that is what makes member-lookup-vs-pointer observable.
  auto dram = Device().getProperty(Path(kMemPerf, {"TransferRate"}));
  ASSERT_TRUE(dram.has_value()) << "MemoryTypes/GDDR6 (Off-chip DRAM)/Performance";
  EXPECT_DOUBLE_EQ(std::get<nxs_double>(AsVariant(*dram)), 16.0);

  auto cache = Device().getProperty(Path(kCachePerf, {"TransferRate"}));
  ASSERT_TRUE(cache.has_value()) << "MemoryTypes/L2 / System Level Cache/Performance";
  EXPECT_DOUBLE_EQ(std::get<nxs_double>(AsVariant(*cache)), 4.0);

  // A name that is not a key resolves to nothing rather than to a neighbour. "GDDR6" is a
  // prefix of the declared "GDDR6 (Off-chip DRAM)" -- and is the key the corpus itself now
  // uses -- so this pins that lookup is exact, not prefix-matched.
  EXPECT_FALSE(Device()
                   .getProperty(std::vector<std::string_view>{
                       "MemorySubsystem", "MemoryTypes", "GDDR6", "Performance",
                       "TransferRate"})
                   .has_value())
      << "a bare 'GDDR6' is not a declared key and must not resolve";
}

TEST_F(PropertyTypes, KeysListsTheMemoryNames) {
  // InfoImpl::getKeys answers only for an object. Under the old array shape this property
  // did not exist; keying the map is what makes the memory names enumerable.
  //
  // The order is the parser's, not the file's: `using json = nlohmann::json` is backed by
  // std::map, so keys come back sorted. The fixture writes the cache first and the DRAM
  // second, and Keys returns them the other way round.
  auto keys = Device().getProperty(Path(kMemTypes, {"Keys"}));
  ASSERT_TRUE(keys.has_value()) << "MemorySubsystem/MemoryTypes/Keys";
  ASSERT_TRUE(std::holds_alternative<knexus::PropStrVec>(AsVariant(*keys)))
      << "Keys came back as variant index " << AsVariant(*keys).index();
  const auto &names = std::get<knexus::PropStrVec>(AsVariant(*keys));
  EXPECT_EQ(names, (knexus::PropStrVec{"GDDR6 (Off-chip DRAM)",
                                       "L2 / System Level Cache"}));

  // The same property on the sibling half of the hierarchy, which was already keyed.
  auto units = Device().getProperty(
      std::vector<std::string_view>{"CoreSubsystem", "UnitTypes", "Keys"});
  ASSERT_TRUE(units.has_value());
  EXPECT_EQ(std::get<knexus::PropStrVec>(AsVariant(*units)),
            (knexus::PropStrVec{"Matrix Unit (FPU)"}));
}

TEST_F(PropertyTypes, IntegerAndStringFieldsKeepTheirDeclaredTypes) {
  auto lane_width = Device().getProperty(Path(kUnitPerf, {"LaneWidth"}));
  ASSERT_TRUE(lane_width.has_value());
  EXPECT_TRUE(std::holds_alternative<nxs_long>(AsVariant(*lane_width)));
  EXPECT_EQ(std::get<nxs_long>(AsVariant(*lane_width)), 32);

  auto block_source = Device().getProperty(Path(kUnitPerf, {"Source"}));
  ASSERT_TRUE(block_source.has_value());
  EXPECT_TRUE(std::holds_alternative<std::string>(AsVariant(*block_source)));
  EXPECT_EQ(std::get<std::string>(AsVariant(*block_source)), "Published");

  for (const auto &pair : std::vector<std::pair<std::string_view, std::string>>{
           {"Unit", "TFLOP/s"}, {"Precision", "BFP8"}, {"Mode", "LoFi"},
           {"Source", "Published"}}) {
    auto value = Device().getProperty(Path(kUnitPerf, {"Throughput", "0", pair.first}));
    ASSERT_TRUE(value.has_value()) << pair.first;
    EXPECT_TRUE(std::holds_alternative<std::string>(AsVariant(*value))) << pair.first;
    EXPECT_EQ(std::get<std::string>(AsVariant(*value)), pair.second) << pair.first;
  }
}

// ---------------------------------------------------------------------------------------
// the control: this is what the declarations are protecting against
// ---------------------------------------------------------------------------------------

TEST_F(PropertyTypes, UndeclaredNumericFieldsTakeTheirTypeFromTheLiteral) {
  ASSERT_EQ(nxsGetPropEnum("NotADeclaredProperty"), NXS_PROPERTY_INVALID);
  ASSERT_EQ(nxsGetPropEnum("AlsoNotADeclaredProperty"), NXS_PROPERTY_INVALID);

  auto whole = Device().getProperty(std::vector<std::string_view>{"ControlGroup", "NotADeclaredProperty"});
  ASSERT_TRUE(whole.has_value());
  EXPECT_TRUE(std::holds_alternative<nxs_long>(AsVariant(*whole)))
      << "an undeclared whole-number literal is expected to come back as an integer -- "
         "that is the defect the KNEXUS_API_PROP rows exist to prevent";

  auto fractional = Device().getProperty(
      std::vector<std::string_view>{"ControlGroup", "AlsoNotADeclaredProperty"});
  ASSERT_TRUE(fractional.has_value());
  EXPECT_TRUE(std::holds_alternative<nxs_double>(AsVariant(*fractional)));

  // The same undeclared field, two device files, two C++ types. A declared field cannot do
  // this -- see WholeNumberLiteralInADeclaredFloatFieldReadsBackAsDouble above.
  EXPECT_NE(AsVariant(*whole).index(), AsVariant(*fractional).index());
}

}  // namespace
