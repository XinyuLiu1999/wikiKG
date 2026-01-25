"""Parquet I/O utilities for batch processing."""

import os
import pyarrow as pa
import pyarrow.dataset as ds
import pyarrow.parquet as pq


class ParquetBatchWriter:
    """Batch writer for Parquet files with streaming support."""

    def __init__(self, path, schema):
        """Initialize the writer.

        Args:
            path: Output file path
            schema: PyArrow schema for the table
        """
        self.path = path
        self.schema = schema
        self._writer = None
        # Handle empty dirname (file in current directory)
        dirname = os.path.dirname(path)
        if dirname:
            os.makedirs(dirname, exist_ok=True)

    def write(self, columns):
        """Write a batch of columns to the file.

        Args:
            columns: Dict mapping column names to lists of values
        """
        table = pa.Table.from_pydict(columns, schema=self.schema)
        if self._writer is None:
            self._writer = pq.ParquetWriter(self.path, self.schema)
        self._writer.write_table(table)

    def close(self):
        """Close the writer and finalize the file."""
        if self._writer is not None:
            self._writer.close()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()
        return False


def iter_parquet_batches(path, columns, batch_size):
    """Iterate over a Parquet file in batches.

    Args:
        path: Path to the Parquet file
        columns: List of column names to read
        batch_size: Number of rows per batch

    Yields:
        PyArrow RecordBatch objects
    """
    dataset = ds.dataset(path, format="parquet")
    for batch in dataset.to_batches(columns=columns, batch_size=batch_size):
        yield batch
