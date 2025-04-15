#include <dwio/nimble/velox/VeloxWriter.h>
#include <folly/init/Init.h>
#include <dwio/nimble/velox/VeloxWriterOptions.h>
#include <velox/type/Type.h>
#include <velox/vector/FlatVector.h>
#include <velox/core/Expressions.h>
#include <velox/vector/VectorSaver.h>
#include <velox/vector/tests/utils/VectorTestBase.h>

using namespace facebook::velox;
using namespace facebook::dwio::common;
using namespace facebook::nimble;

int main(int argc, char** argv) {
  folly::init(&argc, &argv, true);

  // Path to your CSV
  std::string inputCsv = "data.csv";
  std::string outputNimble = "data.nimble";

  // Define the schema explicitly (change this to match your CSV)
  auto rowType = ROW({"id", "name", "price"},
                     {INTEGER(), VARCHAR(), DOUBLE()});

  // Load CSV as RowVector
  auto pool = memory::getDefaultScopedMemoryPool();
  exec::test::CsvReader reader(rowType, inputCsv, *pool, ',', true);
  auto rowVector = reader.read();

  // Configure writer options
  dwio::common::MemorySink sink(outputNimble.c_str());
  VeloxWriterOptions options;
  VeloxWriter writer(rowType, options, &sink);

  // Write
  writer.write(rowVector);
  writer.flush();
  writer.close();

  std::cout << "✅ Wrote Nimble file to: " << outputNimble << std::endl;
  return 0;
}
