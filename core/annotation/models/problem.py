from pydantic import BaseModel
from typing import List
from core.annotation.models.file import File
import yaml
import os
from core.annotation.utils.generate_predictions import generate_predictions
from core.async_chat_clients import AsyncChatClients
from core.data_classes.llm_type import LLMType
from core.annotation.utils.run_shell_command import run_shell_command


class Problem(BaseModel):
    folder_name: str
    paper_tex: str
    context_files: List[File]
    problem_file: File
    test_entry_point: str
    @classmethod
    def parse_problem(cls, pset_dir: str, problem_name: str) -> 'Problem':
        problem_dir = os.path.join(pset_dir, problem_name)

        with open(os.path.join(problem_dir, "paper2code.yaml"), "r") as f:
            problem_yaml = yaml.load(f, Loader=yaml.FullLoader)
        
        with open(os.path.join(problem_dir, problem_yaml["paper_tex"]), "r", encoding="utf-8") as f:
            paper_tex = f.read()
        
        # breakpoint()
        problem_file = File.parse_file(problem_dir, problem_yaml["mask_file_path"])

        # read context files
        context_files = []
        if problem_yaml["context_file_paths"] is not None:
            for context_file_path in problem_yaml["context_file_paths"]:
                context_files.append(File.parse_file(problem_dir, context_file_path))
        return cls(folder_name=problem_name, paper_tex=paper_tex, context_files=context_files, problem_file=problem_file, test_entry_point=problem_yaml["test_entry_point"])



    async def generate_solutions(self, llm_types: List[LLMType], n_completions: int, temperature: float, clients: AsyncChatClients) -> str:
        if len(self.context_files) == 0:
            context_files_str = ''
        else:
            context_files_str = ''.join([file.flatten() for file in self.context_files])
        
        for snippet in self.problem_file.snippets:
            
            # create a new problem file with the snippet masked
            masked_problem_file_str = self.problem_file.mask_given_snippet(snippet, placeholder_lines=None, remove_markers=True)
            masked_code_str = context_files_str + masked_problem_file_str
            
            snippet.predictions = await generate_predictions(
                masked_code_str,
                self.paper_tex,
                llm_types=llm_types, 
                n_completions=n_completions, 
                predictions=snippet.predictions, 
                clients=clients,
                temperature=temperature
            )


    # def test(self, problem_src_dir: str, problem_cache_dir: str) -> List[str]:
        
    #     commands = []
        
    #     for snippet in self.problem_file.snippets:
    #         snippet.test(self.problem_file)
            
    #         # run_shell_command(f"python {problem_src_dir}/test_snippet.py")
            
            