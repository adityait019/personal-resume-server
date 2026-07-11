import pymupdf


class PDFParser:
    """Extracts raw text from a PDF resume, all pages, no side-effect file writes."""

    def __init__(self, pdf_path: str):
        if not isinstance(pdf_path, str):
            raise TypeError("Path must be a string.")
        self.pdf_path = pdf_path

    def extract_text(self) -> str:
        """
        Opens the PDF, concatenates text from every page, and returns it.
        Raises FileNotFoundError / RuntimeError on failure instead of
        printing and swallowing — callers (the ingestion service) need
        to know extraction failed rather than silently getting "".
        """
        doc = None
        try:
            doc = pymupdf.open(self.pdf_path)

            if doc.page_count == 0:
                raise ValueError(f"PDF has no pages: {self.pdf_path}")

            page_texts = []
            for page in doc:
                text = page.get_text("text")
                if text:
                    page_texts.append(text)

            full_text = "\n\n".join(page_texts).strip()

            if not full_text:
                raise ValueError(
                    f"No extractable text found in {self.pdf_path} "
                    "(it may be a scanned/image-only PDF requiring OCR)."
                )

            return full_text

        except FileNotFoundError:
            raise FileNotFoundError(f"File not found: {self.pdf_path}")
        except pymupdf.FileDataError as e:
            raise RuntimeError(f"Could not read PDF (corrupt or unsupported): {e}")
        finally:
            if doc is not None:
                doc.close()


if __name__ == "__main__":
    import sys

    path = sys.argv[1] if len(sys.argv) > 1 else "sample.pdf"
    parser = PDFParser(path)
    print(parser.extract_text())
