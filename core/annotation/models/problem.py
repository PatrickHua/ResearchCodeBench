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
    paper_tex_path: str
    context_files: List[File]
    problem_files: List[File]
    test_entry_point: str
    @classmethod
    def parse_problem(cls, pset_dir: str, problem_name: str, paper_yaml: dict) -> 'Problem':
        problem_dir = os.path.join(pset_dir, problem_name)


        paper_tex_path = os.path.join(problem_dir, 'paper2code_paper.tex')

        if isinstance(paper_yaml["annotated_file_paths"], str):
            annotated_file_paths = [paper_yaml["annotated_file_paths"]]
        elif isinstance(paper_yaml["annotated_file_paths"], list):
            annotated_file_paths = paper_yaml["annotated_file_paths"]
        else:
            raise ValueError(f"Invalid annotated_file_paths: {paper_yaml['annotated_file_paths']}")  # no annotated files is problematic
        
        
        if isinstance(paper_yaml["context_file_paths"], str):
            context_file_paths = [paper_yaml["context_file_paths"]]
        elif isinstance(paper_yaml["context_file_paths"], list):
            context_file_paths = paper_yaml["context_file_paths"]
        else:
            context_file_paths = None  # no context files is fine
        
        
        problem_files = [File.parse_file(problem_dir, annotated_file_path) for annotated_file_path in annotated_file_paths]

        if context_file_paths is not None:
            context_files = [File.parse_file(problem_dir, context_file_path) for context_file_path in context_file_paths]
        else:
            context_files = []


        return cls(folder_name=problem_name, paper_tex_path=paper_tex_path, context_files=context_files, problem_files=problem_files, test_entry_point='paper2code_test.py')



    async def generate_solutions(self, llm_types: List[LLMType], n_completions: int, temperature: float, clients: AsyncChatClients, wo_paper: bool = False) -> str:
        import asyncio
        
        if len(self.context_files) == 0:
            context_files_str = ''
        else:
            context_files_str = ''.join([file.flatten() for file in self.context_files])
        
        with open(self.paper_tex_path, "r", encoding="utf-8") as f:
            paper_tex = f.read()
        
        async def process_snippet(problem_file, snippet):
            # create a new problem file with the snippet masked
            masked_problem_file_str = problem_file.mask_given_snippet(snippet, placeholder_lines=None, remove_markers=True)
            masked_code_str = context_files_str + masked_problem_file_str
            
            snippet.predictions = await generate_predictions(
                masked_code_str,
                paper_tex,
                llm_types=llm_types, 
                n_completions=n_completions, 
                predictions=snippet.predictions, 
                clients=clients,
                temperature=temperature,
                wo_paper=wo_paper
            )
            return snippet
        
        # Gather all snippets across all problem files
        tasks = []
        for problem_file in self.problem_files:
            for snippet in problem_file.snippets:
                tasks.append(process_snippet(problem_file, snippet))
        
        # Run all tasks concurrently
        await asyncio.gather(*tasks)


    # def test(self, problem_src_dir: str, problem_cache_dir: str) -> List[str]:
        
    #     commands = []
        
    #     for snippet in self.problem_file.snippets:
    #         snippet.test(self.problem_file)
            
    #         # run_shell_command(f"python {problem_src_dir}/test_snippet.py")
            
            