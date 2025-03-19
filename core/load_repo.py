
import json
import argparse
import os
from core.data_classes.models import Repository, Paper, Function, Question, Completion, QuestionType, MetricType, ContextType

# python load_repo.py --repo_path '/ccn2/u/tyhua/paper2code/repos.json' --output_path repos_loaded.json

def init_repo(starter_json_file: str) -> list[Repository]:
    """
    Turns:
    
    [
        {
            "name": "mae",
            "description": "description2",
            "url": "https://github.com/facebookresearch/mae",
            "local_path": "/ccn2/u/tyhua/datasets/repos/mae",
            "paper": {
                "title": "title2",
                "authors": ["author3", "author4"],
                "year": 2024,
                "url": null,
                "local_path": null,
                "venue": null
            },
            "function_path": "/ccn2/u/tyhua/paper2code/mae_out.json",  # path to the function json file generated from r2e.
            "novel_fn_ids": ["models_mae.MaskedAutoencoderViT.random_masking", "models_mae.MaskedAutoencoderViT.forward_loss"]  # list of novel function ids chosen by the user.
        },
        ...
    ]
    
    
    into a list of Repository objects:
    [
        {
            "repo_name": "mae",
            "description": "description2",
            "url": "https://github.com/facebookresearch/mae",
            "local_path": "/ccn2/u/tyhua/datasets/repos/mae",
            "paper": {
                "title": "title2",
                "authors": [
                    "author3",
                    "author4"
                ],
                "year": 2024,
                "url": null,
                "local_path": null,
                "venue": null
            },
            "functions": [
                {
                    "function_id": "models_mae.MaskedAutoencoderViT.random_masking",
                    "is_method": true,
                    "description": null,
                    "code": "def random_masking(self, x, mask_ratio):\n    \"\"\"\n   ...",
                    "context_type": "sliced",
                    "context": "```python\n## util/pos_embed.py\nimport numpy as np\n\ndef get_2d_sincos_pos_embed(embed_dim, grid_size, cls_token=False):\n ...",
                    "questions": null,
                    "tests": null,
                    "novelty": null
                },
                ...
            ],
            "novel_fn_ids": ["models_mae.MaskedAutoencoderViT.random_masking", "models_mae.MaskedAutoencoderViT.forward_loss"]
        },
        ...
    ]
    
    """


    with open(starter_json_file, "r") as f:
        repos = json.load(f)

    repo_list = []
    for repo in repos:
        paper = Paper(
            title=repo["paper"]["title"],
            authors=repo["paper"]["authors"],
            year=repo["paper"]["year"],
            url=repo["paper"]["url"],
            local_path=repo["paper"]["local_path"],
            venue=repo["paper"]["venue"],
        )
        functions = []
        with open(repo["function_path"], "r") as f:
            functions_json = json.load(f)

        
        for function_json in functions_json:
            is_method = "method_id" in function_json


            function = Function(
                function_id=function_json["function_id"]['identifier'] if not is_method else function_json["method_id"]['identifier'],
                is_method=is_method,
                description=None,
                code=function_json["function_code"] if not is_method else function_json["method_code"],
                context=function_json["context"]["context"],
                context_type=ContextType(function_json["context"]["context_type"]),
                questions=None,
            )

            functions.append(function)
        repo = Repository(
            repo_name=repo["name"],
            description=repo["description"],
            url=repo["url"],
            local_path=repo["local_path"],
            paper=paper,
            functions=functions,
            novel_fn_ids=repo["novel_fn_ids"],
        )
        repo_list.append(repo)
    return repo_list


def save_repos(repo_list: list[Repository], output_path: str):
    # Check if the output directory exists, create if not
    output_dir = os.path.dirname(output_path)
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    repo_list_dict = [repo.model_dump() for repo in repo_list]
    with open(output_path, "w") as f:
        f.write(json.dumps(repo_list_dict, indent=4))

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo_path", type=str, required=True)
    parser.add_argument("--output_path", type=str, required=False)
    args = parser.parse_args()
    repo_list = init_repo(args.repo_path)
    # print(repo.model_dump_json(indent=4))
    if args.output_path:
        save_repos(repo_list, args.output_path)
    


if __name__ == "__main__":
    main()

















