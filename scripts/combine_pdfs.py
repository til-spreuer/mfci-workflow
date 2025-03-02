from pypdf import PdfReader, PdfWriter
from snakemake.script import Snakemake


def fix_smk() -> Snakemake:
    return snakemake


snakemake = fix_smk()

"""
Combines all PDFs into one PDF
"""

writer = PdfWriter()

for pdf in snakemake.input:
    reader = PdfReader(pdf)
    writer.append(reader)

writer.write(snakemake.output[0])
writer.close()
