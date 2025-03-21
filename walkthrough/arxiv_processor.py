#!/usr/bin/env python3
import os
import shutil
import requests
import tempfile
import re
from pathlib import Path
from rich.console import Console
from rich.panel import Panel
from rich.progress import Progress, SpinnerColumn, TextColumn
from rich.table import Table
from rich.prompt import Prompt, Confirm

console = Console()

class ArxivProcessError(Exception):
    """Custom exception for arXiv processing errors."""
    pass

def flatten_tex_files(source_dir: Path, main_tex: Path) -> str:
    """
    Flatten all tex files in the directory into one, maintaining all content.
    """
    def read_file(file_path: Path) -> str:
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                return f.read()
        except UnicodeDecodeError:
            # Try Latin-1 if UTF-8 fails
            with open(file_path, 'r', encoding='latin-1') as f:
                return f.read()

    def process_content(content: str, base_path: Path) -> str:
        # Handle \input{file} and \include{file}
        pattern = r'\\(?:input|include){([^}]+)}'
        
        def replace_match(match):
            filename = match.group(1)
            # Add .tex extension if not present
            if not filename.endswith('.tex'):
                filename += '.tex'
            
            file_path = base_path / filename
            if file_path.exists():
                sub_content = read_file(file_path)
                # Recursively process the included file
                return process_content(sub_content, file_path.parent)
            else:
                # If exact path not found, try finding the file anywhere in source_dir
                possible_files = list(source_dir.rglob(filename))
                if possible_files:
                    sub_content = read_file(possible_files[0])
                    return process_content(sub_content, possible_files[0].parent)
                console.print(f"[yellow]Warning: Could not find included file: {filename}[/yellow]")
                return match.group(0)
        
        return re.sub(pattern, replace_match, content)

    # Read and process the main file
    content = read_file(main_tex)
    return process_content(content, main_tex.parent)

def verify_tex_content(content: str) -> list[str]:
    """Verify the tex content has expected sections."""
    sections = ['abstract', 'introduction', 'related work', 
                'method', 'experiment', 'conclusion']
    return [s for s in sections if s.lower() not in content.lower()]

def process_arxiv_paper(output_dir: str) -> None:
    """Handle arXiv paper download and processing."""
    while True:
        arxiv_url = console.input("\n[yellow]Please enter the arXiv paper URL or ID:[/yellow] ").strip()
        
        # Extract arxiv ID from URL or direct input
        arxiv_id = arxiv_url
        if "arxiv.org" in arxiv_url:
            match = re.search(r'(\d+\.\d+)', arxiv_url)
            if match:
                arxiv_id = match.group(1)
        
        # Construct download URL
        source_url = f"https://arxiv.org/e-print/{arxiv_id}"
        
        try:
            with console.status("[bold yellow]Downloading arXiv source...", spinner="dots"):
                response = requests.get(source_url)
                if response.status_code != 200:
                    console.print("[red]Error: Could not download arXiv source. Please check the ID/URL.[/red]")
                    continue
                
                # Save to temporary file
                with tempfile.NamedTemporaryFile(suffix='.tar.gz', delete=False) as tmp_file:
                    tmp_file.write(response.content)
                    tmp_path = tmp_file.name

            # Extract files
            temp_extract_dir = tempfile.mkdtemp()
            shutil.unpack_archive(tmp_path, temp_extract_dir)
            os.unlink(tmp_path)

            # Find tex files
            source_dir = Path(temp_extract_dir)
            tex_files = list(source_dir.rglob('*.tex'))
            
            if not tex_files:
                raise ArxivProcessError("No .tex files found in the arXiv source")

            # Combine all tex files
            with console.status("[bold yellow]Processing LaTeX files...", spinner="dots"):
                combined_content = ""
                for tex_file in tex_files:
                    with open(tex_file, 'r', encoding='utf-8') as f:
                        combined_content += f"\n% Content from {tex_file.name}\n"
                        combined_content += f.read() + "\n"
                
                # Verify content
                missing_sections = verify_tex_content(combined_content)
                if missing_sections:
                    console.print(f"[yellow]Warning: File might be missing sections: {', '.join(missing_sections)}[/yellow]")
                    if not Confirm.ask("[yellow]Continue anyway?[/yellow]"):
                        continue

            # Write the combined content
            final_path = os.path.join(output_dir, 'paper2code_paper.tex')
            with open(final_path, 'w', encoding='utf-8') as f:
                f.write(combined_content)
            
            # Cleanup
            shutil.rmtree(temp_extract_dir)
            
            console.print("[green]✓ Successfully processed arXiv paper[/green]")
            break
            
        except Exception as e:
            console.print(f"[red]Error processing arXiv paper: {str(e)}[/red]")
            if not Confirm.ask("[yellow]Would you like to try again?[/yellow]"):
                break

if __name__ == '__main__':
    # Stand-alone testing
    output_dir = "test_output"
    os.makedirs(output_dir, exist_ok=True)
    process_arxiv_paper(output_dir) 