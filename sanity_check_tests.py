import yaml
import os

def main():
    pset_dir = "pset"
    with open(f"{pset_dir}/papers.yaml", "r") as f:
        papers = yaml.safe_load(f)

    # for paper in papers:
    #     print(paper["title"])

    for paper in papers:
        folder_name = paper["id"]
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