from bs4 import BeautifulSoup
# you need BeautifulSoup and lxml packages installed

def delete_clip_path(svg_file):
    
    with open(svg_file, 'r') as file:
        content = file.read()
    soup = BeautifulSoup(content, 'lxml-xml')
    
    clip_paths = soup.find_all('clipPath')
    
    if clip_paths:
        for clip_path in clip_paths:
            # Remove the clip-path attribute from <clipPath> elements
            clip_path.decompose()
            
        print("ClipPath elements deleted.")
    
        modified_content = str(soup)
        
        # Save modified content to a new file
        with open(svg_file, 'w') as output_file:
            output_file.write(modified_content)
        print("File saved.")

    else:
        print("ClipPath elements not found.")


