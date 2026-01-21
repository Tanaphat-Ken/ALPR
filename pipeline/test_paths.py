from pathlib import Path
import pandas as pd

df = pd.read_csv('data/test_images/labels.csv')

def normalize_rel(p: str) -> str:
    # Convert all separators to backslash for Windows, strip leading separator
    return str(p).replace("\\", "/").replace("/", "\\").lstrip("\\")

df['image_rel'] = df['image_name_gray'].astype(str).map(normalize_rel)
df['image_path'] = df['image_rel'].apply(lambda x: Path('data/test_images') / x)
existing = df[df['image_path'].apply(lambda p: p.exists())]

print(f'Total rows: {len(df)}')
print(f'Existing files: {len(existing)}')
if len(existing) > 0:
    print('\nFirst 3 existing paths:')
    for p in existing['image_path'].head(3):
        print(f'  {p}')
else:
    print('\nNo files found. Sample constructed paths:')
    for p in df['image_path'].head(3):
        print(f'  {p} (exists: {p.exists()})')
