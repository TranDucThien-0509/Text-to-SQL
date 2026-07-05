import json
import re


# ════════════════════════════════════════════════════════
# BƯỚC 1: Build schema matching dict (VI → EN)
# ════════════════════════════════════════════════════════

def create_schema_matching_dict(vi_tables_path, en_tables_path):
    """
    Build mapping VI schema → EN schema.
      - Dùng table_names_original cho VI (pipeline sinh SQL từ field này)
      - Dùng column_names_original cho VI (16 DB có _original khác column_names)
      - Sort keys theo độ dài giảm dần khi lưu vào dict (longest-match khi replace)
    """
    with open(vi_tables_path, 'r', encoding='utf-8') as f:
        vi_data = json.load(f)
    with open(en_tables_path, 'r', encoding='utf-8') as f:
        en_data = json.load(f)

    en_db_map = {db['db_id']: db for db in en_data}
    matching_dict = {}

    for vi_db in vi_data:
        db_id = vi_db['db_id']
        if db_id not in en_db_map:
            continue

        en_db = en_db_map[db_id]
        matching_dict[db_id] = {"tables": {}, "columns": {}}

        # --- TABLES ---
        vi_tables = vi_db['table_names_original']
        en_tables = en_db['table_names_original']

        for vi_t, en_t in zip(vi_tables, en_tables):
            vi_t = vi_t.strip()
            en_t = en_t.strip()
            matching_dict[db_id]["tables"][vi_t] = en_t
            vi_t_us = vi_t.replace(' ', '_')
            if vi_t_us != vi_t:
                matching_dict[db_id]["tables"][vi_t_us] = en_t

        # --- COLUMNS ---
        vi_cols = vi_db['column_names_original']
        en_cols = en_db['column_names_original']


        for (_, vi_col), (_, en_col) in zip(vi_cols, en_cols):
            vi_col = vi_col.strip()
            en_col = en_col.strip()
            if vi_col == '*':
                continue
            matching_dict[db_id]["columns"][vi_col] = en_col
            vi_col_us = vi_col.replace(' ', '_')
            if vi_col_us != vi_col:
                matching_dict[db_id]["columns"][vi_col_us] = en_col
                
                
    return matching_dict


# ════════════════════════════════════════════════════════
# BƯỚC 2: Translate VI SQL → EN SQL
# ════════════════════════════════════════════════════════

# Ký tự phân cách hợp lệ trong SQL (không phải chữ cái token)
_SQL_DELIMITERS = r'[\s,.()\[\]=<>!\"\']'


def _make_pattern(vi: str) -> str:
    """
    Tạo regex pattern với SQL-aware boundary thay vì \\w boundary.

    Vấn đề với \\w:
      - Python re với UNICODE: \\w bao gồm cả ký tự tiếng Việt có dấu
      - "sắp xếp khóa học" có space → \\w boundary không protect được token multi-word
      - Ví dụ: (?<!\\w)sắp xếp khóa học(?!\\w) sẽ fail nếu trước "sắp" là ký tự Unicode

    Fix: dùng SQL delimiter (space, dấu câu SQL) làm ranh giới thay vì \\w.
    """
    escaped = re.escape(vi)
    # lookbehind: đứng sau delimiter hoặc đầu chuỗi
    # lookahead:  đứng trước delimiter hoặc cuối chuỗi
    return r'(?:(?<=' + _SQL_DELIMITERS + r')|(?:^|(?<=\.)))' + escaped + r'(?=' + _SQL_DELIMITERS + r'|$)'


def translate_sql(vi_sql: str, db_id: str, matching_dict: dict) -> str:
    """
    Dịch SQL tiếng Việt → tiếng Anh dùng matching_dict.

    Thuật toán:
      1. Gom tất cả mappings (columns + tables), sort theo:
           - Columns trước tables (column "id khóa học" chứa table name "khóa học")
           - Trong mỗi nhóm: longest-first (tránh "khóa học" ăn phần của "sắp xếp khóa học")
      2. Dùng PLACEHOLDER để bảo vệ vùng đã replace:
           - Sau khi replace "sắp xếp khóa học" → __PH0__, token "khóa học"
             không còn match trong chuỗi nữa → không bị double-replace
      3. Cuối cùng restore tất cả placeholder → EN token
    """
    if db_id not in matching_dict:
        return vi_sql

    mapping = matching_dict[db_id]

    # Gom mappings: columns trước, trong mỗi nhóm longest-first
    all_mappings = []
    for vi_col, en_col in mapping["columns"].items():
        if vi_col and vi_col != '*' and vi_col != en_col:
            all_mappings.append((vi_col, en_col, 0))  # priority 0 = column

    for vi_table, en_table in mapping["tables"].items():
        if vi_table and vi_table != en_table:
            all_mappings.append((vi_table, en_table, 1))  # priority 1 = table

    # Sort: (priority ASC, len DESC)
    all_mappings.sort(key=lambda x: -len(x[0])) 

    sql = vi_sql
    placeholders = {}   # ph_token → en_value
    ph_idx = 0

    for vi_tok, en_tok, _ in all_mappings:
        try:
            pattern = _make_pattern(vi_tok)
            if re.search(pattern, sql, flags=re.IGNORECASE):
                ph = f'__PH{ph_idx}__'
                placeholders[ph] = en_tok
                sql = re.sub(pattern, ph, sql, flags=re.IGNORECASE)
                ph_idx += 1
        except re.error:
            # Fallback: plain replace nếu regex lỗi (hiếm gặp)
            if vi_tok in sql:
                ph = f'__PH{ph_idx}__'
                placeholders[ph] = en_tok
                sql = sql.replace(vi_tok, ph)
                ph_idx += 1

    # Restore placeholders → EN tokens
    for ph, en_tok in placeholders.items():
        sql = sql.replace(ph, en_tok)

    return sql


# ════════════════════════════════════════════════════════
# BƯỚC 3: Match câu hỏi VI ↔ EN (positional index)
# ════════════════════════════════════════════════════════

def build_question_alignment(vi_json_path, en_json_path):
    """
    Match câu hỏi VI với câu hỏi EN theo positional index (1-to-1).
    ViSpider được dịch trực tiếp từ Spider → vi[i] ↔ en[i].

    Returns list of:
        {index, db_id, vi_question, en_question, gold_sql}
    """
    with open(vi_json_path, 'r', encoding='utf-8') as f:
        vi_data = json.load(f)
    with open(en_json_path, 'r', encoding='utf-8') as f:
        en_data = json.load(f)

    assert len(vi_data) == len(en_data), (
        f"Số record không khớp: VI={len(vi_data)}, EN={len(en_data)}"
    )

    aligned = []
    mismatches = 0
    for i, (vi, en) in enumerate(zip(vi_data, en_data)):
        if vi['db_id'] != en['db_id']:
            mismatches += 1
        aligned.append({
            "index"      : i,
            "db_id"      : en['db_id'],
            "vi_question": vi['question'],
            "en_question": en['question'],
            "gold_sql"   : en.get('query', ''),
        })

    if mismatches:
        print(f"⚠️  {mismatches} db_id mismatches — kiểm tra lại alignment!")
    else:
        print(f"✓ Question alignment OK: {len(aligned)} records.")

    return aligned


# ════════════════════════════════════════════════════════
# CHẠY THỬ NGHIỆM
# ════════════════════════════════════════════════════════

if __name__ == '__main__':
    VI_FILE = r"C:\Users\Admin\Documents\Uni\DS319\Text-to-SQL\data\word-level\tables.json"
    EN_FILE = r"C:\Users\Admin\Documents\Uni\DS319\Text-to-SQL\data\spider_data\tables_spider.json"

    # --- Build schema map ---
    print("Building schema matching dict...")
    matching_dict = create_schema_matching_dict(VI_FILE, EN_FILE)

    first_db = list(matching_dict.keys())[0]
    print(f"\n🔥 Cấu trúc matching của database '{first_db}':")
    print(json.dumps(matching_dict[first_db], ensure_ascii=False, indent=4))

    with open("schema_matching_map.json", "w", encoding="utf-8") as f:
        json.dump(matching_dict, f, ensure_ascii=False, indent=4)
    print("\n✓ Ghi file 'schema_matching_map.json' thành công!")

    # --- Test translation ---
    print("\n=== Translation Tests ===\n")
    test_cases = [
        # (db_id, vi_sql, expected_en_sql)

        ("architecture",
         'select count ( * ) from kiến_trúc_sư where giới_tính = "female"',
         'select count ( * ) from architect where gender = "female"'),

        ("architecture",
         'select tên , quốc_tịch , id from kiến_trúc_sư where giới_tính = "male" order by tên',
         'select name , nationality , id from architect where gender = "male" order by name'),

        ("architecture",
         'select distinct t1.tên , t1.quốc_tịch from kiến_trúc_sư as t1 join nhà_máy as t2 on t1.id = t2.id kiến_trúc_sư',
         'select distinct t1.name , t1.nationality from architect as t1 join mill as t2 on t1.id = t2.architect_id'),

        ("architecture",
         'select avg ( chiều dài theo feet ) from cầu',
         'select avg ( length_feet ) from bridge'),

        ("course_teach",
         'select t3.tên , t2.khoá_học from sắp_xếp khoá học as t1 join khoá học as t2 on t1.id khoá học = t2.id khoá học join giáo_viên as t3 on t1.id giáo_viên = t3.id giáo_viên',
         'select t3.Name , t2.Course from course_arrange as t1 join Course as t2 on t1.Course_ID = t2.Course_ID join teacher as t3 on t1.Teacher_ID = t3.Teacher_ID'),

        # Test case cho vấn đề "sắp xếp khóa học" vs "khóa học":
        # "sắp xếp khóa học" → "course_arrange" (không bị "khóa học" → "course" ăn mất)
        # Uncomment và điền db_id/token đúng khi có data thực
        # ("course_db",
        #  'select sắp xếp khóa học from khóa học',
        #  'select course_arrange from course'),
    ]

    all_pass = True
    for i, (db_id, vi_sql, expected) in enumerate(test_cases):
        result = translate_sql(vi_sql, db_id, matching_dict)
        ok = result == expected
        status = '✓' if ok else '✗'
        if not ok:
            all_pass = False
        print(f"[{status}] Test {i+1}")
        if not ok:
            print(f"  VI : {vi_sql}")
            print(f"  GOT: {result}")
            print(f"  EXP: {expected}")
        else:
            print(f"  {result[:90]}")
        print()

    print("✓ All tests passed!" if all_pass else "✗ Some tests FAILED.")
