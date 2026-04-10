"""SQL dump parsing utilities for Wikipedia dumps."""

import gzip


def iter_insert_tuples(f, table_name):
    """从已经打开的流（f）中迭代解析 INSERT 语句。
    
    Args:
        f: 已经打开的可迭代对象（可以是 gzip.open 的返回对象，也可以是 subprocess.stdout）
        table_name: 表名
    """
    prefix = f"INSERT INTO `{table_name}` VALUES"
    
    for line in f:
        # 如果是字节流（bytes），先解码成字符串
        if isinstance(line, bytes):
            line = line.decode("utf-8", errors="replace")
            
        if not line.startswith(prefix):
            continue
            
        # 提取 VALUES 之后的部分
        try:
            values_str = line.split("VALUES", 1)[1].strip()
            if values_str.endswith(";"):
                values_str = values_str[:-1]
            
            for tup in _iter_values(values_str):
                yield tup
        except IndexError:
            continue


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
