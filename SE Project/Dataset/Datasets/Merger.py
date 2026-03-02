import pandas as pd
import os
import re

# Define file paths
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
FILES = {
    "srs": "Gitub (srs_dataset).csv",
    "extended": "Kaggle SR Dataset (Vaibhav) SR_Extend.csv", 
    "nfr": "Kaggle SR Dataset (Dumindu Nissanka) NRF.csv",
    "xlsx": "Mendeley SR Dataset (FR_NFR_dataset).xlsx"
}

# Mapping for abbreviations to Full Names
TYPE_MAPPING = {
    'PE': 'Performance',
    'LF': 'Look and Feel',
    'US': 'Usability',
    'A': 'Availability',
    'FT': 'Fault Tolerance',
    'SC': 'Scalability',
    'SE': 'Security',
    'L': 'Legal',
    'F': 'Functional',
    'MN': 'Maintainability',
    'O': 'Operational',
    'PO': 'Portability',
    'FR': 'Functional',
    'NFR': 'Non-Functional',
    'FAU': 'Fault Tolerance',
    'PME': 'Performance',
    'GPS': 'Functional',
    'API': 'Functional',
    'INT': 'Interface' 
}

def load_and_standardize():
    dfs = []
    
    # 1. Load Gitub (srs_dataset).csv
    csv_path = os.path.join(BASE_DIR, FILES["srs"])
    if os.path.exists(csv_path):
        print(f"Loading {FILES['srs']}...")
        df = pd.read_csv(csv_path)
        df.columns = [c.lower() for c in df.columns]
        
        if 'requirement' in df.columns:
            df = df.rename(columns={'requirement': 'text'})
        if 'type' not in df.columns and 'label' in df.columns:
             df = df.rename(columns={'label': 'type'})
             
        # Ensure type is text and mapped
        if 'type' in df.columns:
             df['type'] = df['type'].astype(str).str.strip()
             
             
        df['source'] = 'GitHub SRS'
        dfs.append(df[['text', 'type', 'source']])

    # 2. Load Kaggle SR Dataset (Vaibhav) SR_Extend.csv
    ext_path = os.path.join(BASE_DIR, FILES["extended"])
    if os.path.exists(ext_path):
        print(f"Loading {FILES['extended']}...")
        df = pd.read_csv(ext_path)
        df.columns = [c.lower() for c in df.columns]
        
        if 'requirement' in df.columns:
            df = df.rename(columns={'requirement': 'text'})
        
        # Ensure type is text
        if 'type' in df.columns:
             df['type'] = df['type'].astype(str).str.strip()
             
        df['source'] = 'Kaggle Extended'
        dfs.append(df[['text', 'type', 'source']])

    # 3. Load Kaggle SR Dataset (Dumindu Nissanka) NRF.csv
    nfr_path = os.path.join(BASE_DIR, FILES["nfr"])
    if os.path.exists(nfr_path):
        print(f"Loading {FILES['nfr']}...")
        try:
            # Try reading as standard CSV first
            df = pd.read_csv(nfr_path, on_bad_lines='skip') 
            df.columns = [c.lower() for c in df.columns]
            
            # Guesses for columns
            if 'requirement' in df.columns:
                df = df.rename(columns={'requirement': 'text'})
            if 'class' in df.columns:
                df = df.rename(columns={'class': 'type'})
                
            if 'text' in df.columns and 'type' in df.columns:
                df['source'] = 'Kaggle NFR'
                dfs.append(df[['text', 'type', 'source']])
            else:
                 raise ValueError("Columns not found, trying manual parse")
                 
        except Exception:
            # Fallback for "Type:Requirement" text file disguised as CSV
            rows = []
            with open(nfr_path, 'r', encoding='utf-8', errors='ignore') as f:
                for line in f:
                    line = line.strip()
                    match = re.match(r'^([A-Za-z\s]+)[:|-](.*)', line)
                    if match:
                        rows.append({'type': match.group(1).strip(), 'text': match.group(2).strip()})
            if rows:
                df = pd.DataFrame(rows)
                df['source'] = 'Kaggle NFR Parsed'
                dfs.append(df[['text', 'type', 'source']])

    # 4. Load Mendeley SR Dataset (FR_NFR_dataset).xlsx
    xlsx_path = os.path.join(BASE_DIR, FILES["xlsx"])
    if os.path.exists(xlsx_path):
        print(f"Loading {FILES['xlsx']}...")
        try:
            df = pd.read_excel(xlsx_path)
            df.columns = [c.lower() for c in df.columns]
            
            rename_map = {}
            for c in df.columns:
                if 'req' in c or 'cont' in c or 'text' in c:
                    rename_map[c] = 'text'
                if 'lab' in c or 'type' in c or 'class' in c:
                    rename_map[c] = 'type'
            df = df.rename(columns=rename_map)
            
            if 'text' in df.columns and 'type' in df.columns:
                df['type'] = df['type'].astype(str).str.strip()
                df['source'] = 'Mendeley SRS'
                dfs.append(df[['text', 'type', 'source']])
        except Exception as e:
            print(f"Error loading Excel: {e}")

    # Combine
    if not dfs:
        print("No data found.")
        return None
        
    full_df = pd.concat(dfs, ignore_index=True)
    full_df.dropna(subset=['text'], inplace=True)
    
    # Clean Text
    full_df['text'] = full_df['text'].astype(str).str.strip()
    full_df = full_df[full_df['text'].str.len() > 5] # Remove tiny junk
    
    # Final cleanup of types (Map then Title Case)
    # 1. Normalize to Upper to match mapping and catch variations
    full_df['type'] = full_df['type'].str.upper().str.strip()
    # 2. Apply Mapping
    full_df['type'] = full_df['type'].map(lambda x: TYPE_MAPPING.get(x, x))
    # 3. Title Case
    full_df['type'] = full_df['type'].astype(str).str.title()
    
    # User Request: Rename Columns
    full_df = full_df.rename(columns={
        'text': 'Requirement Text',
        'type': 'Requirement Type',
        'source': 'Source Dataset'
    })
    
    return full_df

if __name__ == "__main__":
    df = load_and_standardize()
    if df is not None:
        # NO HEURISTICS APPLIED
        
        # Save as CSV
        csv_output = os.path.join(BASE_DIR, "combined_srs_dataset.csv")
        df.to_csv(csv_output, index=False)
        print(f"\nProcessing Complete. Saved {len(df)} records to '{csv_output}'.")
        
        # Save as Excel with adjusted column widths
        xlsx_output = os.path.join(BASE_DIR, "combined_srs_dataset.xlsx")
        try:
            with pd.ExcelWriter(xlsx_output, engine='openpyxl') as writer:
                df.to_excel(writer, index=False, sheet_name='Requirements')
                worksheet = writer.sheets['Requirements']
                
                # Adjust column widths
                for col in worksheet.columns:
                    max_length = 0
                    column = col[0].column_letter # Get the column name
                    for cell in col:
                        try:
                            if len(str(cell.value)) > max_length:
                                max_length = len(str(cell.value))
                        except:
                            pass
                    # Set limit to avoiding huge columns for long text
                    adjusted_width = min(max_length + 2, 100) 
                    worksheet.column_dimensions[column].width = adjusted_width
            print(f"Saved Excel file with adjusted columns to '{xlsx_output}'.")
        except Exception as e:
            print(f"Could not save Excel file (missing openpyxl?): {e}")

        print("\nFirst 5 rows:")
        print(df.head())
        print("\nSample of Types found:")
        print(df['Requirement Type'].unique())
    else:
        print("Failed to load any datasets.")
