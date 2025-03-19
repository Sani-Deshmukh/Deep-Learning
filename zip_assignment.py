import os
import zipfile


def main():
    source_directory = "."
    zip_filename = "./submision.zip"
    zip_assignment(source_directory, zip_filename)
    print_zip_tree(zip_filename)
    print(f"Zipped {source_directory} into {zip_filename}")


def zip_assignment(src_dir, zip_name):
    with zipfile.ZipFile(zip_name, "w", zipfile.ZIP_DEFLATED) as zipf:
        for root, dirs, files in os.walk(src_dir):
            dirs[:] = [
                d
                for d in dirs
                if not d.startswith(".")
                and d
                not in (
                    "__pycache__",
                    ".venv",
                    "ref",
                    "gradescope",
                    "dataset",
                    "reference",
                    "data",
                    "results"
                )
                and not d.endswith(".egg-info")
            ]
            files = [
                f
                for f in files
                if (
                    not f.startswith(".") or f in [".python-version"]
                )  # Allow only specific dot files
                and not f.endswith((".pyc", ".pyo", ".egg-info", ".zip"))
                and "ref" not in f  # Exclude all reference files
                and f
                not in [
                    # "zip_assignment.py",
                    "results.json",
                    "requirements_dev.txt",
                    "zip_autograder.py",
                    "uv.lock",
                    "fashion_mnist.zip",
                    "autograder.zip",
                ]
            ]

            for file in files:
                filepath = os.path.join(root, file)
                arcname = os.path.relpath(filepath, src_dir)
                zipf.write(filepath, arcname)


def print_zip_tree(zip_name):
    print("\nContents of the zip file in tree format:")
    with zipfile.ZipFile(zip_name, "r") as zipf:
        file_list = zipf.namelist()
        tree = {}
        for file in file_list:
            parts = file.split("/")
            current = tree
            for part in parts:
                if part not in current:
                    current[part] = {}
                current = current[part]

        def print_tree(tree, prefix=""):
            for key, subtree in tree.items():
                print(f"{prefix}├── {key}")
                print_tree(subtree, prefix + "│   ")

        print_tree(tree)
        print(f"\nTotal files added: {len(file_list)}")


if __name__ == "__main__":
    main()
