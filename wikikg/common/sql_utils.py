"""SQL dump parsing utilities for Wikipedia dumps."""

import gzip


def iter_insert_tuples(path, table_name, encoding="utf-8"):
    """Iterate over INSERT statement tuples from a gzipped SQL dump.

    Args:
        path: Path to the gzipped SQL file
        table_name: Name of the table to extract (e.g., 'page', 'pagelinks')
        encoding: File encoding (default: utf-8)

    Yields:
        Tuples of field values from each INSERT statement
    """
    prefix = f"INSERT INTO `{table_name}` VALUES"
    with gzip.open(path, "rt", encoding=encoding, errors="replace") as f:
        for line in f:
            if not line.startswith(prefix):
                continue
            values_str = line.split("VALUES", 1)[1].strip()
            if values_str.endswith(";"):
                values_str = values_str[:-1]
            for tup in _iter_values(values_str):
                yield tup


def _iter_values(values_str):
    """Parse SQL VALUES clause into individual tuples.

    Handles MySQL escape sequences properly:
    - \\n -> newline
    - \\r -> carriage return
    - \\t -> tab
    - \\0 -> null char
    - \\\\ -> backslash
    - \\' -> single quote
    """
    in_string = False
    escape = False
    in_tuple = False
    field = ""
    fields = []

    for ch in values_str:
        if in_string:
            if escape:
                # Handle MySQL escape sequences
                if ch == 'n':
                    field += '\n'
                elif ch == 'r':
                    field += '\r'
                elif ch == 't':
                    field += '\t'
                elif ch == '0':
                    field += '\0'
                else:
                    # For \\, \', \" and other escaped chars, keep the literal char
                    field += ch
                escape = False
            elif ch == "\\":
                escape = True
            elif ch == "'":
                in_string = False
            else:
                field += ch
            continue

        if ch == "'":
            in_string = True
            continue

        if ch == "(":
            in_tuple = True
            fields = []
            field = ""
            continue

        if ch == ")":
            if in_tuple:
                fields.append(field)
                yield tuple(fields)
                in_tuple = False
                field = ""
            continue

        if ch == ",":
            if in_tuple:
                fields.append(field)
                field = ""
            continue

        if in_tuple and ch not in " \n\r\t":
            field += ch


def to_int(value):
    """Convert SQL value to int, handling NULL."""
    if value == "NULL" or value == "":
        return None
    return int(value)


def to_str(value):
    """Convert SQL value to string, handling NULL."""
    if value == "NULL":
        return None
    return value
