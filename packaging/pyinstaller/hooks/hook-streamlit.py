from PyInstaller.utils.hooks import collect_data_files, collect_submodules, copy_metadata


def include_streamlit_submodule(name):
    return not name.startswith("streamlit.external.langchain")


datas = copy_metadata("streamlit")
datas += collect_data_files("streamlit")

hiddenimports = collect_submodules("streamlit", filter=include_streamlit_submodule)
