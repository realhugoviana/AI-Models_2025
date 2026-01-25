import os
import pandas as pd
import argparse
from pathlib import Path

def get_celebrity_gender(name):
    """
    Determine gender based on actual celebrity knowledge.
    This function uses real knowledge of celebrities from training data.
    """
    # Normalize the name
    name_lower = name.lower().strip()
    
    # Celebrity gender database - using actual knowledge
    # This will be populated based on the celebrities in the dataset
    
    # Since you mentioned the format is generally "pins_THEIRWHOLENAME/FIRSTNAME"
    # I'll extract and determine gender based on my knowledge
    
    # Common celebrity mappings (add more as needed based on your dataset)
    celebrity_genders = {
        'adriana lima': 'F',
        'adriana': 'F',
        'alex': 'M',
        'alexandra daddario': 'F',
        'alexandra': 'F',
        'amber heard': 'F',
        'amber': 'F',
        'amy adams': 'F',
        'amy': 'F',
        'angelina jolie': 'F',
        'angelina': 'F',
        'anne hathaway': 'F',
        'anne': 'F',
        'ben affleck': 'M',
        'ben': 'M',
        'blake lively': 'F',
        'blake': 'F',
        'brad pitt': 'M',
        'brad': 'M',
        'cameron diaz': 'F',
        'cameron': 'F',
        'charlize theron': 'F',
        'charlize': 'F',
        'chris evans': 'M',
        'chris hemsworth': 'M',
        'chris pratt': 'M',
        'chris': 'M',
        'christian bale': 'M',
        'christian': 'M',
        'daniel craig': 'M',
        'daniel': 'M',
        'dwayne johnson': 'M',
        'dwayne': 'M',
        'emma stone': 'F',
        'emma watson': 'F',
        'emma': 'F',
        'gal gadot': 'F',
        'gal': 'F',
        'george clooney': 'M',
        'george': 'M',
        'hugh jackman': 'M',
        'hugh': 'M',
        'jennifer aniston': 'F',
        'jennifer lawrence': 'F',
        'jennifer': 'F',
        'jessica alba': 'F',
        'jessica biel': 'F',
        'jessica': 'F',
        'johnny depp': 'M',
        'johnny': 'M',
        'kate beckinsale': 'F',
        'kate winslet': 'F',
        'kate': 'F',
        'keira knightley': 'F',
        'keira': 'F',
        'kristen stewart': 'F',
        'kristen': 'F',
        'leonardo dicaprio': 'M',
        'leonardo': 'M',
        'margot robbie': 'F',
        'margot': 'F',
        'matt damon': 'M',
        'matt': 'M',
        'megan fox': 'F',
        'megan': 'F',
        'natalie portman': 'F',
        'natalie': 'F',
        'nina dobrev': 'F',
        'nina': 'F',
        'penelope cruz': 'F',
        'penelope': 'F',
        'rachel mcadams': 'F',
        'rachel': 'F',
        'robert downey': 'M',
        'robert': 'M',
        'ryan gosling': 'M',
        'ryan reynolds': 'M',
        'ryan': 'M',
        'scarlett johansson': 'F',
        'scarlett': 'F',
        'tom cruise': 'M',
        'tom hanks': 'M',
        'tom': 'M',
        'will smith': 'M',
        'will': 'M',
        'zendaya': 'F',
        # Additional celebrities from your dataset
        'zac efron': 'M',
        'zac': 'M',
        'inbar lavi': 'F',
        'inbar': 'F',
        'bill gates': 'M',
        'bill': 'M',
        'cristiano ronaldo': 'M',
        'cristiano': 'M',
        'jason momoa': 'M',
        'jason': 'M',
        'grant gustin': 'M',
        'grant': 'M',
        'neil patrick harris': 'M',
        'neil': 'M',
        'rami malek': 'M',
        'rami': 'M',
        'logan lerman': 'M',
        'logan': 'M',
        'stephen amell': 'M',
        'stephen': 'M',
        'henry cavill': 'M',
        'henry cavil': 'M',
        'henry': 'M',
        'miley cyrus': 'F',
        'miley': 'F',
        'lindsey morgan': 'F',
        'lindsey': 'F',
        'richard harmon': 'M',
        'richard': 'M',
        'jimmy fallon': 'M',
        'jimmy': 'M',
        'anthony mackie': 'M',
        'anthony': 'M',
        'keanu reeves': 'M',
        'keanu': 'M',
        'lionel messi': 'M',
        'lionel': 'M',
        'kiernan shipka': 'F',
        'kiernen shipka': 'F',
        'kiernan': 'F',
        'kiernen': 'F',
        'morgan freeman': 'M',
        'morgan': 'M',
        'wentworth miller': 'M',
        'wentworth': 'M',
        'brian j smith': 'M',
        'brian j. smith': 'M',
        'brian': 'M',
        'brenton thwaites': 'M',
        'brenton': 'M',
        'krysten ritter': 'F',
        'krysten': 'F',
        'alvaro morte': 'M',
        'alvaro': 'M',
        'dominic purcell': 'M',
        'dominic': 'M',
        'zoe saldana': 'F',
        'zoe': 'F',
        'taylor swift': 'F',
        'taylor': 'F',
        'elon musk': 'M',
        'elon': 'M',
        'jeremy renner': 'M',
        'jeremy': 'M',
        'jake mcdorman': 'M',
        'jake': 'M',
        'katherine langford': 'F',
        'katherine': 'F',
        'madelaine petsch': 'F',
        'madelaine': 'F',
        'penn badgley': 'M',
        'penn': 'M',
        'katharine mcphee': 'F',
        'katharine': 'F',
        'tuppence middleton': 'F',
        'tuppence': 'F',
        'gwyneth paltrow': 'F',
        'gwyneth': 'F',
        'pedro alonso': 'M',
        'pedro': 'M',
        'avril lavigne': 'F',
        'avril': 'F',
        'bobby morley': 'M',
        'bobby': 'M',
        'barack obama': 'M',
        'barack': 'M',
        'sophie turner': 'F',
        'sophie': 'F',
        'elizabeth olsen': 'F',
        'elizabeth': 'F',
        'mark zuckerberg': 'M',
        'mark': 'M',
        'josh radnor': 'M',
        'josh': 'M',
        'elizabeth lail': 'F',
        'sarah wayne callies': 'F',
        'sarah': 'F',
        'jeff bezos': 'M',
        'jeff': 'M',
        'mark ruffalo': 'M',
        'elliot page': 'M',  # Formerly Ellen Page
        'ellen page': 'M',   # Historical reference
        'marie avgeropoulos': 'F',
        'marie': 'F',
        'lili reinhart': 'F',
        'lili': 'F',
        'millie bobby brown': 'F',
        'millie': 'F',
        'brie larson': 'F',
        'brie': 'F',
    }
    
    # Check direct match first
    if name_lower in celebrity_genders:
        return celebrity_genders[name_lower]
    
    # Try to find partial matches
    for known_name, gender in celebrity_genders.items():
        if known_name in name_lower or name_lower in known_name:
            return gender
    
    # If not found, try to infer from first name patterns
    parts = name_lower.replace('-', ' ').split()
    if parts:
        first_name = parts[0]
        
        # Check if first name is in database
        if first_name in celebrity_genders:
            return celebrity_genders[first_name]
        
        # Common name-based heuristics
        male_indicators = ['mr', 'sir', 'lord', 'king', 'prince']
        female_indicators = ['ms', 'mrs', 'miss', 'lady', 'queen', 'princess']
        
        for indicator in male_indicators:
            if indicator in name_lower:
                return 'M'
        
        for indicator in female_indicators:
            if indicator in name_lower:
                return 'F'
        
        # Name ending heuristics (common patterns)
        if first_name.endswith(('a', 'ia', 'ina', 'ella', 'ette', 'elle', 'ina')):
            return 'F'
    
    return 'Unknown'

def extract_clean_name(folder_name):
    """
    Extract clean celebrity name from folder format.
    Handles formats like 'pins_FIRSTNAME-LASTNAME' or similar.
    """
    # Remove 'pins_' prefix if present
    if folder_name.lower().startswith('pins_'):
        name = folder_name[5:]  # Remove 'pins_'
    else:
        name = folder_name
    
    # Replace hyphens and underscores with spaces
    name = name.replace('-', ' ').replace('_', ' ')
    
    # Remove extra spaces
    name = ' '.join(name.split())
    
    # Ensure proper capitalization from the start
    name = name.lower()
    
    return name

def format_name(name):
    """Format name with proper capitalization"""
    parts = name.split()
    formatted_parts = []
    
    for part in parts:
        # Capitalize each word
        if "'" in part:
            # Handle names with apostrophes
            subparts = part.split("'")
            formatted_part = "'".join([sp.capitalize() for sp in subparts])
        else:
            formatted_part = part.capitalize()
        formatted_parts.append(formatted_part)
    
    return '_'.join(formatted_parts)

def parse_celebrity_name(celeb_folder_name):
    """Parse celebrity folder name and return formatted version with sex"""
    # Extract clean name
    clean_name = extract_clean_name(celeb_folder_name)
    
    # Format name with proper capitalization
    formatted_name = format_name(clean_name)
    
    # Determine gender based on knowledge
    gender = get_celebrity_gender(clean_name)
    
    # Combine: FirstName_MiddleName_LastName_Sex
    result = f"{formatted_name}_{gender}"
    
    return result, gender, clean_name

def rename_folders(csv_path, base_dir, dry_run=True):
    """Rename celebrity folders based on CSV data"""
    
    # Read CSV
    df = pd.read_csv(csv_path)
    
    # Get unique celebrity folder names
    celebrities = df['celebrity'].unique()
    
    print(f"Found {len(celebrities)} unique celebrity folders\n")
    
    renaming_map = {}
    unknown_gender = []
    
    for celeb_folder in celebrities:
        old_name = celeb_folder
        new_name, gender, clean_name = parse_celebrity_name(celeb_folder)
        
        if gender == 'Unknown':
            unknown_gender.append((celeb_folder, clean_name))
        
        old_path = Path(base_dir) / old_name
        new_path = Path(base_dir) / new_name
        
        if old_path.exists():
            renaming_map[old_name] = new_name
            
            gender_indicator = '⚠' if gender == 'Unknown' else '✓'
            
            if dry_run:
                print(f"{gender_indicator} Would rename: {old_name}")
                print(f"            to: {new_name}")
                print(f"      (parsed: {clean_name})")
                print()
            else:
                try:
                    old_path.rename(new_path)
                    print(f"{gender_indicator} Renamed: {old_name} → {new_name}")
                except Exception as e:
                    print(f"✗ Error renaming {old_name}: {e}")
        else:
            print(f"⚠ Folder not found: {old_path}")
    
    if unknown_gender:
        print(f"\n{'='*70}")
        print(f"WARNING: {len(unknown_gender)} celebrities with unknown gender:")
        print(f"{'='*70}")
        for folder_name, clean_name in unknown_gender:
            print(f"  Folder: {folder_name}")
            print(f"  Parsed: {clean_name}")
            print()
        print("Please manually review these and update the script's celebrity_genders dict")
        print("or provide the correct gender information.")
    
    if dry_run:
        print(f"\n{'='*70}")
        print("DRY RUN - No folders were actually renamed")
        print("Run with --execute to apply changes")
        print(f"{'='*70}")
    else:
        print(f"\n✓ Renamed {len(renaming_map)} folders")
    
    return renaming_map

def main():
    parser = argparse.ArgumentParser(
        description='Rename celebrity folders to FirstName_MiddleName_LastName_Sex format'
    )
    parser.add_argument(
        '--csv', 
        type=str, 
        required=True,
        help='Path to the results CSV file'
    )
    parser.add_argument(
        '--dir', 
        type=str, 
        required=True,
        help='Base directory containing celebrity folders'
    )
    parser.add_argument(
        '--execute', 
        action='store_true',
        help='Actually rename folders (default is dry-run)'
    )
    
    args = parser.parse_args()
    
    print(f"CSV file: {args.csv}")
    print(f"Base directory: {args.dir}")
    print(f"Mode: {'EXECUTE' if args.execute else 'DRY RUN'}")
    print(f"{'='*70}\n")
    
    rename_folders(args.csv, args.dir, dry_run=not args.execute)

if __name__ == '__main__':
    main()