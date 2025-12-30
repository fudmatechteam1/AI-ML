"""
Quick setup script to rename your credentials file and create sample developer mappings
"""
import os
import shutil
import pandas as pd
import json

def setup_credentials():
    print("\n" + "="*60)
    print("CREDENTIAL SETUP HELPER")
    print("="*60 + "\n")
    
    #  Rename comprehensive_credentials_database.csv to credentials.csv
    if os.path.exists('comprehensive_credentials_database.csv'):
        if os.path.exists('credentials.csv'):
            print(" credentials.csv already exists - skipping rename")
        else:
            shutil.copy('comprehensive_credentials_database.csv', 'credentials.csv')
            print("Created credentials.csv from comprehensive_credentials_database.csv")
    else:
        print("comprehensive_credentials_database.csv not found!")
        return
    
    #  Load the developer training data if it exists
    if os.path.exists('developer_training_data_with_credentials.csv'):
        df = pd.read_csv('developer_training_data_with_credentials.csv')
        print(f" Found {len(df)} developers in training data")
    elif os.path.exists('github_dataset.csv'):
        # Extract usernames from github_dataset
        df = pd.read_csv('github_dataset.csv')
        df['username'] = df['repositories'].apply(lambda x: x.split('/')[0])
        usernames = df['username'].unique()
        df = pd.DataFrame({'username': usernames})
        print(f" Extracted {len(df)} developers from github_dataset.csv")
    else:
        print(" No dataset found to extract usernames from")
        return
    
    # Step 3: Load credentials database
    creds_df = pd.read_csv('credentials.csv')
    available_creds = creds_df['name'].tolist()
    print(f" Loaded {len(available_creds)} available credentials\n")
    
    #  Generate sample credential assignments
    print("="*60)
    print("GENERATING SAMPLE DEVELOPER CREDENTIALS")
    print("="*60 + "\n")
    
    import random
    random.seed(42)  # For reproducibility
    
    developer_credentials = []
    
    for username in df['username'].head(100):  # Sample first 100 developers
        # Randomly assign 0-4 credentials per developer
        num_creds = random.choices([0, 1, 2, 3, 4], weights=[30, 30, 20, 15, 5])[0]
        
        if num_creds > 0:
            selected_creds = random.sample(available_creds, min(num_creds, len(available_creds)))
            
            # Create credential objects
            cred_objects = []
            for cred_name in selected_creds:
                cred_info = creds_df[creds_df['name'] == cred_name].iloc[0]
                cred_obj = {
                    'name': cred_name,
                    'vendor': cred_info['vendor']
                }
                cred_objects.append(cred_obj)
            
            developer_credentials.append({
                'username': username,
                'credentials': json.dumps(cred_objects)
            })
        else:
            # Developer with no credentials
            developer_credentials.append({
                'username': username,
                'credentials': '[]'
            })
    
    #  Save to CSV
    dev_creds_df = pd.DataFrame(developer_credentials)
    dev_creds_df.to_csv('developer_credentials.csv', index=False)
    
    print(f"Created developer_credentials.csv with {len(dev_creds_df)} developers")
    print(f"  - Developers with credentials: {len([c for c in developer_credentials if c['credentials'] != '[]'])}")
    print(f"  - Developers without credentials: {len([c for c in developer_credentials if c['credentials'] == '[]'])}")
    
    # Show some examples
    print("\n" + "="*60)
    print("SAMPLE DEVELOPER CREDENTIALS (First 5)")
    print("="*60)
    
    for i, row in dev_creds_df.head(5).iterrows():
        creds = json.loads(row['credentials'])
        if creds:
            cred_names = [c['name'] for c in creds]
            print(f"\n{row['username']}:")
            for name in cred_names:
                print(f"  • {name}")
        else:
            print(f"\n{row['username']}: (no credentials)")
    
    print("\n" + "="*60)
    print("SETUP COMPLETE!")
    print("="*60)
    print("\nFiles created:")
    print("  ✓ credentials.csv - Credentials database")
    print("  ✓ developer_credentials.csv - Developer credential mappings")
    print("\nNow run: python model.py")
    print("="*60 + "\n")

if __name__ == "__main__":
    setup_credentials()