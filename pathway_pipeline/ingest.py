# project/pathway_pipeline/ingest.py

import pathway as pw

def get_book_source(data_dir):
    """
    Reads files from the directory.
    Enables 'with_metadata=True' to ensure the 'path' column is available.
    """
    return pw.io.fs.read(
        data_dir,
        format="binary",
        with_metadata=True
    )