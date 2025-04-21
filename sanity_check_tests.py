import yaml
import os
import argparse


os.environ["CUDA_VISIBLE_DEVICES"] = ""
# eomt 
# advantage-alignment
# REPA-E
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--pset_dir", type=str, default="pset")
    parser.add_argument("--folder_name", nargs="+", type=str, default=None)
    args = parser.parse_args()

    pset_dir = args.pset_dir
    with open(f"{pset_dir}/papers.yaml", "r") as f:
        papers = yaml.safe_load(f)

    # for paper in papers:
    #     print(paper["title"])

    for paper in papers:
        folder_name = paper["id"]
        if args.folder_name is not None and folder_name not in args.folder_name:
            continue
        folder_path = f"{pset_dir}/{folder_name}"
        if not os.path.exists(folder_path):
            print(f"Folder {folder_path} does not exist")
            continue

        # annotated_file_paths = paper["annotated_file_paths"]
        print(folder_name)
        os.system(f'cd {folder_path} && python paper2code_test.py')
        print("--------------------------------")
        

if __name__ == "__main__":
    main()