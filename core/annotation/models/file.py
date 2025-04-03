from pydantic import BaseModel
from typing import List, Optional
from core.annotation.models.code import Code
from core.annotation.models.snippet import Snippet
import re
import sys
from typing import Union
import os
START_PATTERN = re.compile(r'^\s*#\s*<paper2code\s+name="([^"]+)">\s*$')
END_PATTERN   = re.compile(r'^\s*#\s*</paper2code\s+name="([^"]+)">\s*$')


class File(BaseModel):
    rel_path: str
    code: Code
    snippets: Optional[List[Snippet]] = None

    @classmethod
    def parse_file(cls, problem_dir: str, rel_path: str) -> 'File':
        # 1. Validate the file
        with open(os.path.join(problem_dir, rel_path), 'r', encoding='utf-8') as f:
            code = Code(f.read())

            
        if not validate_paper2code_file(code.lines, enforce_unique_names=False):
            print("File failed validation. Please fix the errors above.")
            sys.exit(1)

        snippets: List[Snippet] = []

        for i, line in enumerate(code.lines):  # for each line in the file
            start_match = START_PATTERN.match(line)
            end_match = None

            if start_match:
                snippet_lines = []
                for j in range(i+1, len(code.lines)):
                    end_match = END_PATTERN.match(code.lines[j])
                    if end_match and start_match.group(1) == end_match.group(1):
                        break
                    snippet_lines.append(code.lines[j])
                snippet_name = start_match.group(1)
                assert end_match is not None
                new_snippet = Snippet(name=snippet_name, start_line=i, end_line=j, code=Code(snippet_lines))
                snippets.append(new_snippet)

        return cls(rel_path=rel_path, code=code, snippets=snippets)

    def flatten(self)->str:
        """
        Returns a single string of all the code in all the files.
        """
        result = []
        result.append(f"## {self.rel_path}\n\n")
        result.extend(self.code.lines)
        result.append("\n\n")
        return ''.join(result)

    def mask_given_snippet(self, snippet: Snippet, placeholder_lines: List[str]=None, remove_markers: bool = True) -> str:
        # masked_files = []
        # for snippet in self.snippets:
        masked_snippet_file_str = self.build_replaced_code(
            placeholder_lines=placeholder_lines,  # default placeholder lines
            remove_markers=remove_markers,
            snippet=snippet
        )
        return masked_snippet_file_str
            
            

    def build_replaced_code(self, placeholder_lines: List[str]=None, remove_markers: bool = True, snippet: Snippet=None) -> str:
        """
        Returns a string for the entire file with 'snippet' removed:
        - Lines [start_line..end_line] replaced by a placeholder block
        - That placeholder block is indented to match snippet's indentation
        - Optionally remove ALL snippet markers from the replaced file
        """
        from core.annotation.models.file import remove_all_paper2code_markers

        # How many code lines are inside the snippet
        num_lines = len(snippet.code.get_code_lines())

        # Find the snippet's indentation
        snippet_indent = snippet.get_indentation()

        # Construct placeholder lines, matching indentation
        if placeholder_lines is None:
            placeholder_lines = [
                f"# TODO: Implement block \"{snippet.name}\"",
                f"# Approximately {num_lines} line(s) of code."
            ]

        indented_placeholder_lines = [f"{snippet_indent}{line}\n" for line in placeholder_lines]

        # Get lines before and after the snippet
        # lines_before, lines_after = snippet.get_non_snippet_lines(self.code.lines)
        lines_before = self.code.lines[:snippet.start_line]
        lines_after = self.code.lines[snippet.end_line+1:]
        
        
        # Combine everything
        replaced_lines = []
        replaced_lines.extend(lines_before)
        replaced_lines.extend(indented_placeholder_lines)
        replaced_lines.extend(lines_after)

        # Optionally remove ALL snippet markers from the replaced code
        replaced_code = Code(replaced_lines)
        if remove_markers:
            replaced_code = remove_all_paper2code_markers(replaced_code)

        return str(replaced_code)





















def validate_paper2code_file(filepath: Union[str, List[str]], enforce_unique_names: bool = False) -> bool:
    """
    Validates that <paper2code> tags in `filepath` are well-formed:
      - Start tags are '# <paper2code name="XYZ">'
      - End tags are   '# </paper2code name="XYZ">'
      - They match in proper order
      - If 'enforce_unique_names' is True, snippet names must be unique
      - Start and end tags must have the same indentation level

    Returns True if valid, otherwise prints errors/warnings and returns False.
    """
    if isinstance(filepath, str):
        with open(filepath, 'r', encoding='utf-8') as f:
            lines = f.readlines()
    else:
        lines = filepath

    stack = []  # Will store tuples of (snippet_name, indentation)
    used_names = set()
    valid = True

    for i, line in enumerate(lines):
        start_match = START_PATTERN.match(line)
        end_match   = END_PATTERN.match(line)

        if start_match:
            snippet_name = start_match.group(1)
            # Get the indentation (whitespace before the #)
            indentation = len(line) - len(line.lstrip())
            
            if enforce_unique_names:
                if snippet_name in used_names:
                    print(f"[Line {i+1}] Error: Duplicate snippet name '{snippet_name}' found.")
                    valid = False
                else:
                    used_names.add(snippet_name)
            stack.append((snippet_name, indentation))

        elif end_match:
            snippet_name = end_match.group(1)
            # Get the indentation of the end tag
            end_indentation = len(line) - len(line.lstrip())
            
            if not stack:
                print(f"[Line {i+1}] Error: Closing tag for '{snippet_name}' but no snippet is open.")
                valid = False
            else:
                top_name, start_indentation = stack.pop()
                if top_name != snippet_name:
                    print(f"[Line {i+1}] Error: Closing tag for '{snippet_name}' "
                          f"does not match open snippet '{top_name}'.")
                    valid = False
                
                # Check if indentations match
                if start_indentation != end_indentation:
                    print(f"[Line {i+1}] Error: Indentation mismatch for '{snippet_name}'. "
                          f"Start tag had {start_indentation} spaces, end tag has {end_indentation}.")
                    valid = False
        # else: normal line -> no checks

    if stack:
        # Some snippet(s) not closed
        for unclosed_name, _ in stack:
            print(f"Error: Unclosed snippet '{unclosed_name}'.")
        valid = False

    return valid

def remove_all_paper2code_markers(code: Code) -> Code:
    """
    Removes ALL lines that match the start or end snippet markers
    from a block of code (so none remain).
    """
    filtered_lines = []
    for line in code.lines:
        if START_PATTERN.match(line) or END_PATTERN.match(line):
            continue  # skip this marker line
        filtered_lines.append(line)
    return Code(filtered_lines)
