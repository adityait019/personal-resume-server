import pymupdf  




class PDFParser:
    def __init__(self, pdf_path):
        self.pdf_path = pdf_path
        self.doc = None

    def open_pdf(self):
        try:
            if not isinstance(self.pdf_path, str):
                raise TypeError("Path must be a string.")

            self.doc = pymupdf.open(self.pdf_path)   # open the document
            print("Pages:", self.doc.page_count)

            # Read text from the first page
            if self.doc.page_count > 0:
                page = self.doc[0]
                text = page.get_text("text")
                if not isinstance(text, str):
                    text = str(text)
                print("First page text:")
                print(text)
                with open("./resume/output.txt", "w", encoding="utf-8") as f:
                    f.write(text)
                print("Text from the first page has been written to output.txt")
                
            else:
                print("The document has no pages.")

        except FileNotFoundError:
            print("File not found:", self.pdf_path)
        except Exception as e:
            print("Error:", e)
        finally:
            if self.doc is not None:
                self.doc.close()
# Example usage
if __name__ == "__main__":

    pdf_path = r"C:\Users\adity\project\personal-resume-server\resume\aditya_resume (2).pdf"
    pdf_parser = PDFParser(pdf_path)

    pdf_parser.open_pdf()