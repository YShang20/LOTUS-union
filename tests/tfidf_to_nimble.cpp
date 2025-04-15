

// ----- 2. minimal C++ (pseudo‑code) -----
#include <nimble/Writer.h>
#include <velox/parquet/reader/ParquetReader.h>

int main() {
  auto arrowTable = velox::parquet::read("/Users/yolandazhou/Documents/untitled_folder/CSE_584/lotus-584/tests/tfidf_sparse.parquet");
  nimble::TableWriter writer("tfidf.nimble");
  writer.addColumn("row", arrowTable->column(0));
  writer.addColumn("col", arrowTable->column(1));
  writer.addColumn("val", arrowTable->column(2));
  writer.finish();
}
