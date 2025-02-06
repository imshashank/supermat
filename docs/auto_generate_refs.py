"""Generate the code reference pages and navigation."""

from pathlib import Path

import mkdocs_gen_files

nav = mkdocs_gen_files.Nav()  # pyright: ignore[reportPrivateImportUsage]

pkg = "supermat"

for path in sorted(Path(pkg).rglob("*.py")):
    module_path = path.relative_to(pkg).with_suffix("")
    doc_path = path.relative_to(pkg).with_suffix(".md")
    full_doc_path = Path("reference", doc_path)

    parts = list(module_path.parts)

    if "tests" in parts:
        continue
    elif parts[0] == "__init__":
        parts = []
        doc_path = doc_path.with_name("index.md")
        full_doc_path = full_doc_path.with_name("index.md")
    elif parts[-1] == "__init__":
        parts = parts[:-1]
        doc_path = doc_path.with_name("index.md")
        full_doc_path = full_doc_path.with_name("index.md")
    elif parts[-1] == "__main__":
        continue

    if Path("docs", full_doc_path).is_file():
        mkdocs_gen_files.log.warn(f"{full_doc_path} already exists! Skipping!")
        continue

    parts.insert(0, pkg)

    with mkdocs_gen_files.open(full_doc_path, "w", encoding="utf-8") as fd:
        dot_path = ".".join(parts)
        fd.write(f"::: {dot_path}")

    mkdocs_gen_files.set_edit_path(full_doc_path, path)

with mkdocs_gen_files.open("index.md", "w", encoding="utf-8") as fp:
    with Path("README.md").open("r", encoding="utf-8") as source:
        readme_text = source.read()
        readme_text = readme_text.replace("docs/assets/", "assets/")
        fp.write(readme_text)

if Path("LICENSE.txt").exists():
    with mkdocs_gen_files.open("LICENSE.md", "w", encoding="utf-8") as fp:
        with Path("LICENSE.txt").open("r", encoding="utf-8") as source:
            fp.writelines(source.readlines())
