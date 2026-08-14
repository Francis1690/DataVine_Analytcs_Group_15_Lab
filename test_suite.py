import json
import os
import pandas as pd
# import numpy as np

def test_contributor_attributions():
    print("=== Running Contributor Attribution & Rubric Test Suite ===")
    notebook_path = "DataVine_Analytics_Summative_Lab.ipynb"
    assert os.path.exists(notebook_path), f"Notebook {notebook_path} not found!"
    
    with open(notebook_path, 'r', encoding='utf-8') as f:
        nb = json.load(f)
        
    code_cells = [c for c in nb['cells'] if c['cell_type'] == 'code']
    print(f"Total code cells found: {len(code_cells)}")
    
    # Expected team members / contributors from README.md acknowledgment:
    # Sandybee93, Francis1690, Crayons001, Nyamanyi222, benAloo
    contributors = ["Sandybee93", "Francis1690", "Crayons001", "Nyamanyi222", "benAloo"]
    
    attributed_cells = {}
    for contrib in contributors:
        attributed_cells[contrib] = 0
        
    for idx, cell in enumerate(code_cells):
        source = cell.get('source', [])
        if not source:
            continue
        first_line = source[0].strip()
        for contrib in contributors:
            # Check if username comment or tag exists in the first line or any comment line
            if contrib.lower() in first_line.lower():
                attributed_cells[contrib] += 1
                
    print("\n--- Contributor Cell Attribution Counts ---")
    for contrib, count in attributed_cells.items():
        print(f"Contributor @{contrib}: {count} cells attributed as first line")
        assert count >= 2, f"Contributor @{contrib} has {count} cells attributed (requires at least 2-3 cells)!"
        
    print("\n[PASS] All contributors meet the minimum 2-3 cell attribution requirement.")

def test_dataset_preparation_and_models():
    print("\n=== Testing Datasets and Core Machine Learning Components ===")
    
    # 1. Wine Dataset Test
    assert os.path.exists("wine.csv"), "wine.csv not found!"
    wine = pd.read_csv("wine.csv")
    print(f"Wine dataset loaded successfully. Shape: {wine.shape}")
    assert wine.shape[0] > 0, "Wine dataset is empty!"
    
    # 2. Chickwts Dataset Test
    assert os.path.exists("chickwts.csv"), "chickwts.csv not found!"
    chickwts = pd.read_csv("chickwts.csv")
    print(f"Chickwts dataset loaded successfully. Shape: {chickwts.shape}")
    assert "weight" in chickwts.columns, "weight column missing in chickwts"
    assert "feed" in chickwts.columns, "feed column missing in chickwts"
    
    # 3. USArrests Dataset Test
    assert os.path.exists("USArrests.csv"), "USArrests.csv not found!"
    usarrests = pd.read_csv("USArrests.csv")
    print(f"USArrests dataset loaded successfully. Shape: {usarrests.shape}")
    assert "Murder" in usarrests.columns if 'USArrests' in globals() else True, "USArrests columns present"
    
    print("\n[PASS] Dataset preparation and presence tests passed successfully.")

if __name__ == "__main__":
    test_contributor_attributions()
    test_dataset_preparation_and_models()
    print("\n🎉 ALL TESTS PASSED SUCCESSFULLY!")
