#!/usr/bin/env python3
import os
import shutil
import yaml
import requests
import tempfile
import zipfile
import re
import argparse
from pathlib import Path
from rich.console import Console
from rich.panel import Panel
from rich.progress import Progress, SpinnerColumn, TextColumn
from rich.table import Table
from rich.prompt import Prompt, Confirm
from rich import print as rprint
from arxiv_processor import process_arxiv_paper

console = Console()

class PaperCodeError(Exception):
    """Custom exception for paper2code setup errors."""
    pass

def download_repo(url: str, target_dir: str) -> bool:
    """Download repository from URL to target directory. Returns True if new download, False if skipped."""
    if os.path.exists(target_dir):
        console.print(f"\n[bold yellow]Directory already exists: {target_dir}[/bold yellow]")
        console.print("[yellow]Skipping to arXiv paper processing...[/yellow]")
        return False
    
    if "github.com" in url:
        parts = url.rstrip('/').split('github.com/')
        if len(parts) == 2:
            owner_repo = parts[1]
            api_url = f"https://api.github.com/repos/{owner_repo}/zipball/main"
            
            with Progress(
                SpinnerColumn(),
                TextColumn("[progress.description]{task.description}"),
                transient=True,
            ) as progress:
                progress.add_task("Downloading repository...", total=None)
                response = requests.get(api_url)
                
                if response.status_code == 200:
                    with tempfile.NamedTemporaryFile() as tmp_file:
                        tmp_file.write(response.content)
                        tmp_file.seek(0)
                        
                        progress.add_task("Extracting files...", total=None)
                        with zipfile.ZipFile(tmp_file.name) as zip_ref:
                            root_dir = zip_ref.namelist()[0].split('/')[0]
                            zip_ref.extractall(tempfile.gettempdir())
                            
                            temp_extract_path = os.path.join(tempfile.gettempdir(), root_dir)
                            shutil.copytree(temp_extract_path, target_dir)
                            shutil.rmtree(temp_extract_path)
                    return True
                else:
                    raise PaperCodeError(f"[red]Failed to download repository: {response.status_code}[/red]")
    else:
        raise PaperCodeError("[red]Only GitHub URLs are supported at the moment[/red]")

def create_yaml_config(repo_dir: str) -> None:
    """Create paper2code.yaml with the exact configuration from instructions."""
    config_content = """mask_file_path: models/mar.py  # the file where your core code is

context_file_paths:  # these files will be added to the context of the llm. write null if the mask file is self-contained.
  - models/pixelloss.py  # other files that are used in the code
  - util/visualize.py  # other contextual files that are needed for implementing the core code.

test_entry_point: paper2code_test.py  # will run this entry point script to check if the code is correct. paper2code_test.py is a conventional name.
# Use this name unless you have a good reason to change it.

paper_tex: paper2code_paper.tex  # the tex file for the paper.
# name it paper2code_paper.tex unless you have a good reason to change it."""
    
    yaml_path = os.path.join(repo_dir, 'paper2code.yaml')
    with open(yaml_path, 'w') as f:
        f.write(config_content)

def show_instructions() -> None:
    """Show the setup instructions panel."""
    instructions = """
[bold]Paper2Code Setup Instructions:[/bold]

1. [cyan]Create Project Structure[/cyan]
   • Download the repository (do not clone it)
   • Copy the repository to the [yellow]pset[/yellow] folder
   
2. [cyan]Create Configuration[/cyan]
   • Inside the repository, create [yellow]paper2code.yaml[/yellow] with fields:
     - mask_file_path: Path to core code file
     - context_file_paths: List of contextual files (or null)
     - test_entry_point: paper2code_test.py
     - paper_tex: paper2code_paper.tex

3. [cyan]Setup Paper Content[/cyan]
   • Download the paper's tex file from arXiv
   • Create [yellow]paper2code_paper.tex[/yellow]
   • If multiple tex files, merge them:
     [dim]latexpand main_file.tex > merged.tex[/dim]

4. [cyan]Create Test Script[/cyan]
   • Create [yellow]paper2code_test.py[/yellow]
   • Implement tests to verify generated code
   • Ensure tests can run with CPU only

5. [cyan]Final Steps[/cyan]
   • Create requirements.txt or environment.yaml
   • Remove unnecessary files (__pycache__, .git/, etc.)
   • Verify all files are present and tests pass
    """
    console.print(Panel(instructions, 
                       title="[bold blue]Paper2Code Setup Guide[/bold blue]",
                       border_style="blue",
                       expand=False))

def setup_paper2code() -> None:
    """Main function to set up paper2code project with interactive prompts."""
    console.print(Panel.fit(
        "[bold blue]Welcome to paper2code setup![/bold blue]",
        border_style="blue"
    ))
    
    # Get repository URL
    while True:
        repo_url = console.input("\n[yellow]Please enter the GitHub repository URL:[/yellow] ").strip()
        if repo_url.lower() == "skip":
            repo_name = console.input("\n[yellow]Please enter the name of the repository you are working with:[/yellow] ").strip()
            if not os.path.exists("pset/"+repo_name):
                console.print(f"[red]Error: Repository {repo_name} does not exist[/red]")
                continue
            output_dir = "pset/"+repo_name
            break
        elif repo_url.startswith("https://github.com/"):
            output_dir = os.path.join('pset', Path(repo_url).stem)
            break
        else:
            console.print("[red]Error: Please enter a valid GitHub URL (https://github.com/...) or 'skip'[/red]")
    
    try:
        # Step 1 & 2: Download repository and create yaml (if not skipped)
        if repo_url.lower() != "skip":
            is_new_download = download_repo(repo_url, output_dir)
            if is_new_download:
                create_yaml_config(output_dir)
        
        # Step 3: Process arXiv paper
        process_arxiv_paper(output_dir)
        
        # Create summary table
        table = Table(title="Setup Summary", show_header=True, header_style="bold magenta")
        table.add_column("Component", style="cyan")
        table.add_column("Status", style="green")
        table.add_column("Location", style="yellow")
        
        if repo_url.lower() != "skip":
            if is_new_download:
                table.add_row("Repository", "✅ Downloaded", output_dir)
                table.add_row("paper2code.yaml", "✅ Created", f"{output_dir}/paper2code.yaml")
            else:
                table.add_row("Repository", "⚠️ Using Existing", output_dir)
        else:
            table.add_row("Repository", "⏩ Skipped", output_dir)
        
        table.add_row("paper2code_paper.tex", "✅ Created", f"{output_dir}/paper2code_paper.tex")
        
        console.print("\n", table)
        
        # Next steps panel
        next_steps = """
[bold]Next Steps:[/bold]

1. Verify the paths in [cyan]paper2code.yaml[/cyan] match your repository structure
2. Verify the content of [cyan]paper2code_paper.tex[/cyan]
3. Create [cyan]paper2code_test.py[/cyan] for testing
        """
        console.print(Panel(next_steps, title="[bold green]Setup Complete!", border_style="green"))
        
    except PaperCodeError as e:
        console.print(Panel(str(e), title="[bold red]Setup Failed", border_style="red"))
        return
    except Exception as e:
        console.print(Panel(f"[red]Unexpected error: {str(e)}[/red]", 
                          title="[bold red]Setup Failed", 
                          border_style="red"))
        return

def main():
    setup_paper2code()

if __name__ == '__main__':
    main()