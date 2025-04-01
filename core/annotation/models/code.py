from pydantic import BaseModel
from typing import List, Union

class Code(BaseModel):
    lines: List[str]
    
    def __str__(self):
        return "\n".join(self.lines)

    def __init__(self, lines: Union[str, List[str]], **data):
        # Process the input
        if isinstance(lines, str):
            processed_lines = lines.splitlines()
        else:
            processed_lines = lines
        
        # Pass the processed data to Pydantic's constructor
        super().__init__(lines=processed_lines, **data)
    def get_code_lines(self) -> List[str]:
        """
        Returns a list of lines that contain actual code.
        Excludes:
        - Empty lines
        - Comment-only lines (starting with #)
        Preserves code that appears before comments on the same line.
        """
        code_lines = []
        for line in self.lines:
            line_content = line.strip()
            
            # Skip empty lines
            if not line_content:
                continue
                
            # Skip comment-only lines
            if line_content.startswith('#'):
                continue
                
            # For lines with code and comments, keep only the code part
            if '#' in line:
                code_part = line.split('#', 1)[0].rstrip()
                # Only add if there's actual code (not just whitespace) before the comment
                if code_part.strip():
                    code_lines.append(line)
            else:
                code_lines.append(line)
            
        return code_lines

if __name__ == "__main__":
    # Example 1
    code = Code("""
                
                
                
    # This is a comment
    print("Hello, world!") # This is another comment
    """)
    print(code)
    print(code.get_code_lines())