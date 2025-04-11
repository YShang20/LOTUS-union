import pandas as pd
import csv

def process_csv(file_path, max_rows=100):
    """
    Process a CSV file containing ltable and rtable columns.
    If the last column ('gold') is 1, keep only ltable content.
    Otherwise, keep both ltable and rtable content.
    Only use the first max_rows rows from each table.
    Stack the tables vertically and remove duplicates.
    
    Args:
        file_path (str): Path to the CSV file
        max_rows (int): Maximum number of rows to use from each table
        
    Returns:
        pd.DataFrame: Processed dataframe
    """
    # Read the CSV file with proper handling of quoted fields
    # and skip the first 5 lines which contain metadata
    df = pd.read_csv(
        file_path, 
        skiprows=5,
        
    )
    
    # Handle case where parsing might have failed
    if df.shape[0] == 0:
        # Try a more lenient approach
        with open(file_path, 'r', encoding='utf-8') as f:
            # Skip metadata lines
            for _ in range(5):
                next(f)
                
            # Read header
            header = next(f).strip().split(',')
            
            # Read the rest of the data
            data = []
            for line in f:
                if line.strip():
                    data.append(line.strip().split(',', len(header) - 1))
            
            # Create dataframe
            df = pd.DataFrame(data, columns=header)
    
    # Ensure the gold column is properly formatted
    if 'gold' in df.columns:
        # Try to convert gold to numeric, coercing errors to NaN
        df['gold'] = pd.to_numeric(df['gold'], errors='coerce')
        # Fill NaN with 0 (assuming if it's not explicitly 1, we want both tables)
        df['gold'] = df['gold'].fillna(0)
    else:
        # If gold column is missing, assume we want both tables
        df['gold'] = 0
    
    # Identify columns from each table
    ltable_cols = [col for col in df.columns if col.startswith('ltable.')]
    rtable_cols = [col for col in df.columns if col.startswith('rtable.')]
    
    # Create separate dataframes for each table
    table1 = pd.DataFrame()
    table2 = pd.DataFrame()
    
    # Remove the 'ltable.' prefix from column names
    for col in ltable_cols:
        table1[col.replace('ltable.', '')] = df[col]
    
    # Remove the 'rtable.' prefix from column names
    for col in rtable_cols:
        table2[col.replace('rtable.', '')] = df[col]
    
    # Limit each table to max_rows
    table1 = table1.head(max_rows)
    table2 = table2.head(max_rows)
    
    # Process based on the 'gold' column
    result_dfs = []
    
    # Use only the rows corresponding to the limited tables
    for idx, row in df.iloc[:max_rows].iterrows():
        try:
            # Check if it's a number and equal to 1
            if pd.notna(row['gold']) and float(row['gold']) == 1:
                # Keep only table1 for this row
                result_dfs.append(table1.iloc[[idx if idx < len(table1) else -1]])
            else:
                # Keep both tables for this row
                result_dfs.append(table1.iloc[[idx if idx < len(table1) else -1]])
                result_dfs.append(table2.iloc[[idx if idx < len(table2) else -1]])
        except (ValueError, TypeError, IndexError):
            # If we can't convert to float or index is out of bounds, default to keeping both if possible
            if idx < len(table1):
                result_dfs.append(table1.iloc[[idx]])
            if idx < len(table2):
                result_dfs.append(table2.iloc[[idx]])
    
    # Concatenate all resulting rows
    result = pd.concat(result_dfs, ignore_index=True)
    
    # Remove duplicate rows (all columns must match)
    result = result.drop_duplicates()
    
    return result

def main():
    try:
        file_path = "/Users/yolandazhou/Documents/untitled_folder/CSE_584/lotus/labeled_data_cite.csv"  # Update with your file path
        max_rows = 400  
        print(f"Processing file: {file_path}")
        print(f"Using only the first {max_rows} rows from each table")
        
        # Read original CSV to get some stats
        try:
            original_df = pd.read_csv(file_path, skiprows=5,on_bad_lines='skip')
            print(len(original_df))

            #print(original_df.iloc[255:258])
            df = pd.read_csv(file_path, quotechar='"', escapechar='\\')
            ltable_cols = [col for col in original_df.columns if col.startswith('ltable.')]
            rtable_cols = [col for col in original_df.columns if col.startswith('rtable.')]
            
            print(f"Original CSV has {len(original_df)} rows")
            print(f"Table 1 has {len(ltable_cols)} columns starting with 'ltable.'")
            print(f"Table 2 has {len(rtable_cols)} columns starting with 'rtable.'")
        except Exception as e:
            print(f"Could not read original CSV for stats: {e}")
        
        # Process the CSV
        result = process_csv(file_path, max_rows)
        
        # Display information about the result
        print("\nProcessing complete:")
        print(f"After processing, the result has {len(result)} rows")
        print(f"Columns in the result: {', '.join(result.columns)}")
        
        # Check for any null values in the result
        null_counts = result.isnull().sum().sum()
        if null_counts > 0:
            print(f"Warning: Result contains {null_counts} null values")
        
        # Save the result to a new CSV
        output_file = 'processed_result_1417.csv'
        result.to_csv(output_file, index=False)
        print(f"Result saved to '{output_file}'")
        
    except Exception as e:
        print(f"Error in main: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()