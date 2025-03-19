from pydantic import BaseModel
from typing import List
from core_annotation.models.file import File
from core_annotation.models.code import Code
from typing import Dict
import os
import yaml


class Repo(BaseModel):
    files: List[File]
    paper_tex: str
    mask_file_path: str
    context_file_paths: List[str]
    test_script_path: str
    

    @staticmethod
    def flatten_files(files: List[File]) -> List[str]:
        """
        Returns a single string of all the code in all the files.
        """
        result = []
        for file in files:
            result.append(f"## {file.rel_path}\n\n")
            result.extend(file.code.lines)
            result.append("\n\n")
        return result



    def flatten_relevant_files(self, masked_files: List[File]) -> List[str]:
        """
        Returns a single string of all the code in all the files.
        """
        result = []
        mask_file_code = None
        for file in masked_files:
            if file.rel_path in self.context_file_paths:
                result.append(f"## {file.rel_path}\n\n")
                result.extend(file.code.lines)
                result.append("\n\n")
            elif file.rel_path == self.mask_file_path:
                mask_file_code = file.code
        
        assert mask_file_code is not None
        result.append(f"## {self.mask_file_path}\n\n")
        result.extend(mask_file_code.lines)
        result.append("\n\n")
        return result

    @staticmethod
    def collect_file_contents(src_folder: str) -> Dict[str, Code]:
        """
        Recursively find all .py files in src_folder, concatenate them, and return the result as a string.
        """
        result = {}
        for root, _, files in os.walk(src_folder):
            for file in sorted(files):  # Sorting ensures consistent order
                if file.endswith(".py") and not file.startswith("paper2code"):
                    file_path = os.path.join(root, file)
                    rel_path = os.path.relpath(file_path, src_folder)  # Get relative path

                    # Write file contents
                    with open(file_path, "r", encoding="utf-8") as in_f:
                        result[rel_path] = Code(in_f.read())

        return result


    @classmethod
    def parse_repo(cls, repo_dir: str) -> 'Repo':
        collected_contents = Repo.collect_file_contents(repo_dir)

        files = []
        for rel_path, code in collected_contents.items():  # for each file
            file = File.parse_file(rel_path, code)
            files.append(file)
        
        with open(os.path.join(repo_dir, "paper2code_paper.tex"), "r", encoding="utf-8") as f:
            paper_tex = f.read()
            
        with open(os.path.join(repo_dir, "paper2code.yaml"), "r", encoding="utf-8") as f:
            paper2code_yaml = yaml.load(f, Loader=yaml.FullLoader)
            mask_file_path = paper2code_yaml["mask_file_path"]
            context_file_paths = paper2code_yaml["context_file_paths"]
            test_script_path = paper2code_yaml["test_script_path"]
            
        # breakpoint()
        assert os.path.exists(os.path.join(repo_dir, mask_file_path))
        assert all([os.path.exists(os.path.join(repo_dir, path)) for path in context_file_paths])
        assert os.path.exists(os.path.join(repo_dir, test_script_path))

        return cls(files=files, paper_tex=paper_tex, mask_file_path=mask_file_path, context_file_paths=context_file_paths, test_script_path=test_script_path)


