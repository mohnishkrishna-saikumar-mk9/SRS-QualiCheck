"""
Script to create consolidated dataset CSV from ALL sources including Excel
Parses: ARFF files, CSV files, and Excel files
Creates: srs_dataset.csv with columns: text, type
"""
import csv
import re
import os
import glob

try:
    import pandas as pd
    HAS_PANDAS = True
except ImportError:
    HAS_PANDAS = False
    print("Warning: pandas not available, skipping Excel files")

# Type mappings
TYPE_MAP = {
    'F': 'Functional',
    'FR': 'Functional',
    'A': 'Availability',
    'L': 'Legal',
    'LF': 'Look & Feel',
    'MN': 'Maintainability',
    'O': 'Operational',
    'PE': 'Performance',
    'SC': 'Scalability',
    'SE': 'Security',
    'US': 'Usability',
    'FT': 'Fault Tolerance',
    'PO': 'Portability',
    'NFR': 'Non-Functional'
}

def parse_arff_complete(file_path):
    """Parse ARFF file - both data and comment-embedded requirements"""
    requirements = []
    
    try:
        with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
            content = f.read()
            
            # Pattern 1: Data lines - project_id,'text',type
            matches = re.findall(r"\d+,'(.+?)',([A-Z]{1,3})", content)
            for text, req_type in matches:
                text = text.replace("\\'", "'").replace("\\92", "'")
                if len(text) > 15:
                    requirements.append({
                        'text': text.strip(),
                        'type': req_type,
                        'type_full': TYPE_MAP.get(req_type, req_type)
                    })
            
            # Pattern 2: Comment-embedded - % 1,9, The system shall..., F
            comment_matches = re.findall(r'%\s*\d+,\s*\d+,\s*(.+?),\s*([A-Z]{1,3})\s*$', content, re.MULTILINE)
            for text, req_type in comment_matches:
                text = text.strip()
                if len(text) > 15:
                    requirements.append({
                        'text': text,
                        'type': req_type,
                        'type_full': TYPE_MAP.get(req_type, req_type)
                    })
                    
    except Exception as e:
        print(f"  Error: {e}")
    
    return requirements

def parse_excel(file_path):
    """Parse Excel file for requirements"""
    requirements = []
    
    if not HAS_PANDAS:
        return requirements
    
    try:
        # Read all sheets
        xlsx = pd.ExcelFile(file_path)
        print(f"  Sheets: {xlsx.sheet_names}")
        
        for sheet in xlsx.sheet_names:
            df = pd.read_excel(xlsx, sheet_name=sheet)
            print(f"    {sheet}: {len(df)} rows, columns: {list(df.columns)}")
            
            # Try to find text and type columns
            text_col = None
            type_col = None
            
            for col in df.columns:
                col_lower = str(col).lower()
                if 'requirement' in col_lower or 'text' in col_lower or 'description' in col_lower:
                    text_col = col
                elif 'type' in col_lower or 'class' in col_lower or 'category' in col_lower:
                    type_col = col
            
            # If no specific columns found, use first text-like column
            if text_col is None:
                for col in df.columns:
                    if df[col].dtype == 'object':
                        text_col = col
                        break
            
            if text_col:
                for idx, row in df.iterrows():
                    text = str(row[text_col]).strip() if pd.notna(row[text_col]) else ""
                    req_type = str(row[type_col]).strip().upper() if type_col and pd.notna(row.get(type_col)) else "F"
                    
                    # Clean up type
                    if req_type in ['FUNCTIONAL', 'FR']:
                        req_type = 'F'
                    elif req_type in ['NON-FUNCTIONAL', 'NFR', 'NF']:
                        req_type = 'NFR'
                    
                    if len(text) > 15 and text.lower() not in ['nan', 'none', '']:
                        requirements.append({
                            'text': text,
                            'type': req_type if req_type in TYPE_MAP else 'F',
                            'type_full': TYPE_MAP.get(req_type, 'Functional')
                        })
                        
    except Exception as e:
        print(f"  Error reading Excel: {e}")
    
    return requirements

def main():
    base_dir = r"d:\6th Semester\SRS Quality Score"
    
    all_requirements = []
    
    # 1. Find ALL ARFF files
    arff_files = glob.glob(os.path.join(base_dir, "0-datasets", "**", "*.arff"), recursive=True)
    print(f"=== ARFF Files ({len(arff_files)}) ===")
    
    for arff_file in arff_files:
        filename = os.path.basename(arff_file)
        reqs = parse_arff_complete(arff_file)
        all_requirements.extend(reqs)
        print(f"  {filename}: {len(reqs)}")
    
    arff_count = len(all_requirements)
    print(f"Subtotal from ARFF: {arff_count}\n")
    
    # 2. CSV files
    print("=== CSV Files ===")
    csv_files = [
        (os.path.join(base_dir, "nfr.csv"), 0),
        (os.path.join(base_dir, "software_requirements_extended.csv"), 1),
    ]
    
    for csv_file, text_col in csv_files:
        if os.path.exists(csv_file):
            filename = os.path.basename(csv_file)
            count = 0
            try:
                with open(csv_file, 'r', encoding='utf-8', errors='ignore') as f:
                    reader = csv.reader(f)
                    next(reader, None)
                    for row in reader:
                        if len(row) > text_col:
                            text = row[text_col].strip()
                            type_match = re.match(r'^([A-Z]{1,3})\s+(.+)', text)
                            if type_match:
                                req_type = type_match.group(1)
                                text = type_match.group(2)
                            else:
                                req_type = 'F'
                            
                            if len(text) > 20:
                                all_requirements.append({
                                    'text': text,
                                    'type': req_type,
                                    'type_full': TYPE_MAP.get(req_type, req_type)
                                })
                                count += 1
            except Exception as e:
                print(f"  Error: {e}")
            print(f"  {filename}: {count}")
    
    csv_count = len(all_requirements) - arff_count
    print(f"Subtotal from CSV: {csv_count}\n")
    
    # 3. Excel files
    print("=== Excel Files ===")
    excel_files = glob.glob(os.path.join(base_dir, "*.xlsx"))
    excel_files += glob.glob(os.path.join(base_dir, "0-datasets", "**", "*.xlsx"), recursive=True)
    
    for excel_file in excel_files:
        filename = os.path.basename(excel_file)
        print(f"  {filename}:")
        reqs = parse_excel(excel_file)
        all_requirements.extend(reqs)
        print(f"    Total: {len(reqs)}")
    
    excel_count = len(all_requirements) - arff_count - csv_count
    print(f"Subtotal from Excel: {excel_count}\n")
    
    # Total before dedup
    print(f"TOTAL BEFORE DEDUP: {len(all_requirements)}")
    
    # Remove duplicates
    seen = set()
    unique = []
    for req in all_requirements:
        key = req['text'][:80].lower().strip()
        if key not in seen:
            seen.add(key)
            unique.append(req)
    
    print(f"AFTER DEDUP: {len(unique)} unique requirements\n")
    
    # Write CSV
    output_file = os.path.join(base_dir, "srs_dataset.csv")
    with open(output_file, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(['text', 'type', 'type_full'])
        for req in unique:
            writer.writerow([req['text'], req['type'], req['type_full']])
    
    # Summary
    print("="*55)
    print("FINAL DATASET SUMMARY")
    print("="*55)
    
    type_counts = {}
    for req in unique:
        t = req['type']
        type_counts[t] = type_counts.get(t, 0) + 1
    
    for t, count in sorted(type_counts.items(), key=lambda x: -x[1]):
        name = TYPE_MAP.get(t, t)
        print(f"  {t:5} ({name:20}): {count:5}")
    
    print("="*55)
    print(f"  TOTAL UNIQUE REQUIREMENTS: {len(unique)}")
    print("="*55)
    print(f"\nOutput: {output_file}")

if __name__ == '__main__':
    main()
